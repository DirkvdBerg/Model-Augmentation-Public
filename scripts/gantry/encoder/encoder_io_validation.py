"""
encoder_io_validation.py
------------------------
Three-way encoder pre-training comparison:

  Training A: State MSE with TRUE system states (x_logical from MATLAB data)
  Training B: Output prediction loss (n-step RK4 rollout through baseline dynamics)
  Training C: State MSE with BASELINE-SIMULATED states (linearized at Y_op=0)

Training A validates that the encoder architecture can learn the state map.
Training B tests whether output-prediction loss alone produces good states.
Training C tests what happens with approximate (linearized) state targets,
relevant when true states are unavailable (e.g. on hardware).

THEORY: Hoekstra 2026 Eq. 35 -- data-based encoder init via state MSE loss.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/encoder_io_validation.py
"""

import os
import sys
import copy
import json
import time
from datetime import datetime
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
    lr=1e-4,           # HEURISTIC: matches encoder_io_comparison.py
    epochs=200,
    batch_size=256,
    n_steps=20,        # HEURISTIC: matches encoder_io_comparison.py (5 ms at 4 kHz)
    up_sample=2,       # HEURISTIC: matches encoder_io_comparison.py (2 RK4 substeps)
    scheduler_patience=20,  # HEURISTIC: reduce lr if val loss stalls for 20 epochs
    scheduler_factor=0.5,
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

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder', run_id)
os.makedirs(OUT_DIR, exist_ok=True)

# Logical coordinate names
STATE_NAMES = ['X', 'theta', 'Y', 'dX', 'dtheta', 'dY']
PHYS_UNITS = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']

# Stage coordinate names (output space)
STAGE_NAMES = ['x1', 'x2', 'Y']
STAGE_UNITS = ['m', 'm', 'm']

# P^T maps logical positions to stage positions: y_stage = P^T @ q_logical
P_np = P.numpy().astype(DTYPE_NP)
PT = P_np.T  # (3, 3)


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
    central finite-diff for velocities."""
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
    """Compute normalization constants from training data."""
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
# Create windowed data for n-step output prediction
# =============================================================================

def create_io_windows(u, y, norm, n_steps):
    """Create (u_hist, y_hist, u_future, y_future) windows for n-step training.

    u_hist, y_hist: encoder input (normalized).
    u_future: (M, n_steps, nu) normalized inputs for RK4 rollout.
    y_future: (M, n_steps, ny) raw (physical) output targets.
    """
    na_total = na + na_right  # 26
    nb_total = nb + nb_right  # 26
    history = max(na_total, nb_total)
    N = u.shape[0]
    M = N - history - n_steps  # need n_steps future samples after encoder time k

    u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (y - norm['y0']) / norm['ystd']

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    u_future = np.zeros((M, n_steps, nu), dtype=DTYPE_NP)
    y_future = np.zeros((M, n_steps, ny), dtype=DTYPE_NP)

    for i in range(M):
        k = history + i
        # Encoder sees history up to and including time k
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]
        y_hist[i] = y_norm[k - na_total + 1: k + 1]
        # Future: u[k], u[k+1], ..., u[k+n_steps-1] (normalized, for RK4 input)
        u_future[i] = u_norm[k: k + n_steps]
        # Future: y[k+1], y[k+2], ..., y[k+n_steps] (raw physical, for loss target)
        y_future[i] = y[k + 1: k + 1 + n_steps]

    return u_hist, y_hist, u_future, y_future


# =============================================================================
# Create state-target windows (for secondary state-quality evaluation)
# =============================================================================

def create_state_windows(u, y, x_target, norm):
    """Create (u_hist, y_hist, x_target_norm) windows for state evaluation.
    Same windowing as encoder_baseline_standalone.py."""
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    N = u.shape[0]
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
# Analytical baseline states (P_inv + backward finite-diff)
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
# Baseline-simulated states (linearized model at Y_op=0)
# =============================================================================

# HEURISTIC: discard initial transient from baseline simulation.
# theta mode settling time is ~1.3s = 5200 samples at 4 kHz.
N_TRANSIENT = 5000

def simulate_baseline_states(u_stage, dt):
    """Forward-simulate linearized baseline: x[k+1] = Ad @ x[k] + Bd @ u[k].

    Uses the linearized gantry model at Y_op=0. Input u_stage is raw stage
    forces (NOT normalized). Returns x in logical coordinates [X,theta,Y,dX,dtheta,dY].
    """
    Ad, Bd, _, _ = gantry_linearize_and_discretize(dt)
    N = u_stage.shape[0]
    x = np.zeros((N, NX_PHYS), dtype=np.float64)
    for k in range(N - 1):
        x[k + 1] = Ad @ x[k] + Bd @ u_stage[k]
    return x.astype(DTYPE_NP)


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
# Build state block (shared between training and evaluation)
# =============================================================================

def build_state_block(norm):
    """Build Gantry_State_Block for n-step rollout."""
    state_block = Gantry_State_Block(
        Y_op=None,        # LPV mode: Y read from state
        std_x=norm['std_x'],
        std_u=norm['std_u'],
        x_mean=norm['x_mean'],
        u_mean=norm['u_mean'],
        Ts=TS_NEW,
        up_sample=HP['up_sample'],  # HEURISTIC: 2 matches encoder_io_comparison.py
    ).to(DTYPE_PT)
    return state_block


# =============================================================================
# N-step rollout (shared forward pass for training and evaluation)
# =============================================================================

def nstep_rollout(encoder, state_block, u_hist_t, y_hist_t, u_future_t,
                  std_x_t, x_mean_t, Cd_t):
    """Roll encoder states forward n_steps through RK4, return y_hat per step.

    Returns: y_hat_steps (batch, n_steps, 3) in physical stage coordinates.
    """
    # Encoder: (batch, 6) normalized states at time k
    x = encoder(u_hist_t, y_hist_t)       # (batch, 6)
    x = x.unsqueeze(-1)                    # (batch, 6, 1)

    n_steps = u_future_t.shape[1]
    y_hat_list = []

    for step in range(n_steps):
        # u at time k+step, normalized
        u_step = u_future_t[:, step, :].unsqueeze(-1)   # (batch, 3, 1)
        z = torch.cat([x, u_step], dim=1)               # (batch, 9, 1)

        # RK4 step: x[k+step] -> x[k+step+1], all normalized
        x = state_block.nonlinear_function(z)            # (batch, 6, 1)

        # Denormalize to physical, apply output matrix
        x_phys = x * std_x_t + x_mean_t                 # (batch, 6, 1)
        y_hat = (Cd_t @ x_phys).squeeze(-1)              # (batch, 3)
        y_hat_list.append(y_hat)

    return torch.stack(y_hat_list, dim=1)  # (batch, n_steps, 3)


# =============================================================================
# Output-prediction loss
# =============================================================================

def output_prediction_loss(y_hat_steps, y_future_t, ystd_t):
    """MSE on normalized output prediction, averaged over steps and channels.

    y_hat_steps: (batch, n_steps, 3) predicted outputs (physical).
    y_future_t:  (batch, n_steps, 3) measured outputs (physical).
    ystd_t:      (3,) output std for normalization.
    """
    err_norm = (y_hat_steps - y_future_t) / ystd_t   # (batch, n_steps, 3)
    return torch.mean(err_norm ** 2)


# =============================================================================
# Evaluation metrics
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


def compute_output_nrms_per_horizon(y_hat_steps, y_future):
    """Per-channel NRMS at each prediction horizon.

    y_hat_steps: (M, n_steps, 3).
    y_future:    (M, n_steps, 3).
    Returns: (n_steps, 3) NRMS array.
    """
    n_steps = y_hat_steps.shape[1]
    nrms = np.zeros((n_steps, 3), dtype=np.float64)
    for h in range(n_steps):
        nrms[h] = compute_nrms(y_hat_steps[:, h, :], y_future[:, h, :])
    return nrms


def evaluate_encoder_states(encoder, u_hist, y_hist, x_target):
    """Evaluate state reconstruction quality (secondary metric)."""
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
# N-step evaluation (full validation set, no gradient)
# =============================================================================

def evaluate_nstep(encoder, state_block, u_hist, y_hist, u_future, y_future,
                   norm):
    """Evaluate n-step output prediction on validation data.
    Returns y_hat (M, n_steps, 3) and per-horizon NRMS (n_steps, 3)."""
    std_x_t = torch.tensor(norm['std_x'].astype(DTYPE_NP), dtype=DTYPE_PT)
    x_mean_t = torch.tensor(norm['x_mean'].astype(DTYPE_NP), dtype=DTYPE_PT)
    Cd_t = Cd.to(DTYPE_PT)

    encoder.eval()
    state_block.eval()
    with torch.no_grad():
        y_hat_t = nstep_rollout(
            encoder, state_block,
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
            torch.tensor(u_future, dtype=DTYPE_PT),
            std_x_t, x_mean_t, Cd_t,
        )
    y_hat = y_hat_t.numpy()
    nrms_per_h = compute_output_nrms_per_horizon(y_hat, y_future)
    return y_hat, nrms_per_h


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
    ax.semilogy(train_losses, label='train (output MSE)', linewidth=0.8)
    ax.semilogy(val_losses, label='val (output MSE)', linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized output MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Encoder I/O validation: training loss ({HP["n_steps"]}-step)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_output_prediction(y_hat, y_target, nrms, horizon_label, out_path):
    """Time-domain overlay: y_hat vs y_measured for 3 output channels."""
    T = min(2000, y_hat.shape[0])
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        rms = compute_rms_error(y_hat[:, i:i+1], y_target[:, i:i+1])[0]
        rms_str = format_rms_unit(rms, STAGE_UNITS[i])
        ax.plot(t, y_target[:T, i], 'k-', linewidth=0.8, label='measured')
        ax.plot(t, y_hat[:T, i], 'r--', linewidth=0.8,
                label=f'predicted (NRMS={nrms[i]:.2e}, RMS={rms_str})')
        ax.set_ylabel(f'{STAGE_NAMES[i]} [{STAGE_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Encoder I/O validation: {horizon_label} output prediction', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_output_error(y_hat_enc, y_hat_ana, y_target, out_path):
    """Output prediction error: encoder vs analytical at a given horizon."""
    T = min(2000, y_hat_enc.shape[0])
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        enc_err = y_hat_enc[:T, i] - y_target[:T, i]
        ana_err = y_hat_ana[:T, i] - y_target[:T, i]
        ax.plot(t, enc_err, 'r-', linewidth=0.6, label='encoder error')
        ax.plot(t, ana_err, 'b-', linewidth=0.6, alpha=0.7, label='analytical error')
        ax.set_ylabel(f'{STAGE_NAMES[i]} error [{STAGE_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    n_steps = HP['n_steps']
    fig.suptitle(f'Encoder I/O validation: {n_steps}-step output error comparison', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_horizon_nrms(nrms_enc, nrms_ana, nrms_init, out_path):
    """NRMS vs prediction horizon: encoder vs analytical vs init."""
    n_steps = nrms_enc.shape[0]
    horizons = np.arange(1, n_steps + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.semilogy(horizons, nrms_enc[:, i], 'r-o', markersize=3, linewidth=1.0,
                    label='encoder (trained)')
        ax.semilogy(horizons, nrms_ana[:, i], 'b-s', markersize=3, linewidth=1.0,
                    label='analytical (P_inv + bwd diff)')
        ax.semilogy(horizons, nrms_init[:, i], '--', color='tab:orange', markersize=3,
                    linewidth=1.0, label='model-based init')
        ax.set_ylabel(f'{STAGE_NAMES[i]} NRMS')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Prediction horizon [steps]')
    fig.suptitle('Encoder I/O validation: NRMS vs prediction horizon', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_state_comparison(x_enc, x_ana, x_target, nrms_enc, nrms_ana,
                          rms_enc, rms_ana, out_path):
    """State reconstruction: encoder vs analytical vs target (logical coords)."""
    n_states = len(STATE_NAMES)
    T = min(2000, len(x_enc))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(n_states, 1, figsize=(14, 2.5 * n_states), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_target[:T, i], 'k-', linewidth=0.8, label='target')
        rms_enc_str = format_rms_unit(rms_enc[i], PHYS_UNITS[i])
        rms_ana_str = format_rms_unit(rms_ana[i], PHYS_UNITS[i])
        ax.plot(t, x_enc[:T, i], 'r--', linewidth=0.8,
                label=f'encoder (NRMS={nrms_enc[i]:.2e}, RMS={rms_enc_str})')
        ax.plot(t, x_ana[:T, i], 'b:', linewidth=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e}, RMS={rms_ana_str})')
        ax.set_ylabel(f'{STATE_NAMES[i]} [{PHYS_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Encoder I/O validation: state reconstruction (logical coords)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_nrms_bar(nrms_before, nrms_after, nrms_ana, out_path):
    """Bar chart: state NRMS per channel (init / trained / analytical)."""
    n = len(STATE_NAMES)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, nrms_before, w, label='model-based init', color='tab:orange', alpha=0.8)
    ax.bar(x, nrms_after, w, label='after I/O training', color='tab:red', alpha=0.8)
    ax.bar(x + w, nrms_ana, w, label='analytical baseline', color='tab:blue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Encoder I/O validation: state NRMS comparison')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    n_steps = HP['n_steps']

    print('=' * 70)
    print(f'Encoder I/O validation: {n_steps}-step output prediction (no MSD)')
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
    print(f'  Max velocity NRMS (Python vs MATLAB): {np.max(vel_nrms):.4e}')
    if np.max(vel_nrms) < 0.01:
        print('  OK: Python velocities match MATLAB within 1%.')
    else:
        print('  WARNING: Python velocities differ from MATLAB significantly.')

    # Python-computed state targets for secondary evaluation
    val_x_target = compute_velocities_from_positions(val_y, norm['P_inv_T'])

    # --- Create windowed datasets ---
    print(f'\nCreating {n_steps}-step windows...')
    train_windows = []
    for u, y, _ in train_data:
        uh, yh, uf, yf = create_io_windows(u, y, norm, n_steps)
        train_windows.append((uh, yh, uf, yf))

    u_train = np.concatenate([w[0] for w in train_windows])
    y_train = np.concatenate([w[1] for w in train_windows])
    uf_train = np.concatenate([w[2] for w in train_windows])
    yf_train = np.concatenate([w[3] for w in train_windows])
    print(f'  Training windows: {len(u_train)}')

    uh_val, yh_val, uf_val, yf_val = create_io_windows(val_u, val_y, norm, n_steps)
    print(f'  Validation windows: {len(uh_val)}')

    # State windows for STATE MSE training (using x_logical from .mat as targets)
    print('\nCreating state windows for state MSE training...')
    train_state_windows = []
    for u, y, x_log in train_data:
        uh, yh, xt = create_state_windows(u, y, x_log, norm)
        train_state_windows.append((uh, yh, xt))

    u_train_state = np.concatenate([w[0] for w in train_state_windows])
    y_train_state = np.concatenate([w[1] for w in train_state_windows])
    x_train_target = np.concatenate([w[2] for w in train_state_windows])
    print(f'  State training windows: {len(u_train_state)}')

    # State-evaluation windows (no future needed, covers more samples)
    sx_uh_val, sx_yh_val, sx_xt_val = create_state_windows(
        val_u, val_y, val_x_target, norm)
    print(f'  State validation windows: {len(sx_uh_val)}')

    # Baseline-simulated state windows (Training C: linearized baseline at Y_op=0)
    print(f'\nSimulating baseline model (linearized at Y_op=0)...')
    train_bsim_windows = []
    for u, y, _ in train_data:
        x_bsim = simulate_baseline_states(u, TS_NEW)
        # Discard initial transient, replace with steady-state value
        x_bsim[:N_TRANSIENT] = x_bsim[N_TRANSIENT]
        uh, yh, xt = create_state_windows(u, y, x_bsim, norm)
        train_bsim_windows.append((uh, yh, xt))

    u_train_bsim = np.concatenate([w[0] for w in train_bsim_windows])
    y_train_bsim = np.concatenate([w[1] for w in train_bsim_windows])
    x_train_bsim = np.concatenate([w[2] for w in train_bsim_windows])
    print(f'  Baseline-sim training windows: {len(u_train_bsim)}')

    # Validation baseline-simulated states (for Training C val loss)
    x_bsim_val = simulate_baseline_states(val_u, TS_NEW)
    x_bsim_val[:N_TRANSIENT] = x_bsim_val[N_TRANSIENT]
    bsim_uh_val, bsim_yh_val, bsim_xt_val = create_state_windows(
        val_u, val_y, x_bsim_val, norm)
    print(f'  Baseline-sim validation windows: {len(bsim_uh_val)}')

    # Check: how much do baseline-simulated states differ from true states?
    x_bsim_val_full = simulate_baseline_states(val_u, TS_NEW)
    bsim_vs_true_nrms = compute_nrms(
        x_bsim_val_full[N_TRANSIENT:], val_x_logical[N_TRANSIENT:])
    print(f'\n  Baseline-simulated vs true states (after transient):')
    print(f'  {"State":<8s}  {"NRMS":>10s}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {bsim_vs_true_nrms[i]:>10.4e}')

    # --- Build encoders and state block ---
    print('\nBuilding encoders (model-based init)...')
    encoder = build_encoder(norm)
    encoder_smse = copy.deepcopy(encoder)   # Training A: state MSE (true states)
    encoder_bsim = copy.deepcopy(encoder)   # Training C: state MSE (baseline-simulated)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'  Parameters per encoder: {n_params}')

    state_block = build_state_block(norm)

    # Precompute torch tensors for rollout
    std_x_t = torch.tensor(norm['std_x'].astype(DTYPE_NP), dtype=DTYPE_PT)
    x_mean_t = torch.tensor(norm['x_mean'].astype(DTYPE_NP), dtype=DTYPE_PT)
    Cd_t = Cd.to(DTYPE_PT)
    ystd_t = torch.tensor(norm['ystd'], dtype=DTYPE_PT)

    # --- Evaluate BEFORE training (state quality) ---
    print('\n--- BEFORE training: state quality ---')
    x_hat_before, nrms_before = evaluate_encoder_states(
        encoder, sx_uh_val, sx_yh_val, sx_xt_val)
    x_hat_ana = compute_analytical_baseline(val_y, norm)
    nrms_ana = compute_nrms(x_hat_ana, sx_xt_val)

    print(f'  {"State":<8s}  {"Model-based init":>18s}  {"Analytical":>14s}')
    print(f'  {"-"*8}  {"-"*18}  {"-"*14}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {nrms_before[i]:>18.4e}  {nrms_ana[i]:>14.4e}')

    # --- Evaluate BEFORE training (output prediction) ---
    print(f'\n--- BEFORE training: {n_steps}-step output prediction ---')
    y_hat_before, nrms_out_before = evaluate_nstep(
        encoder, state_block, uh_val, yh_val, uf_val, yf_val, norm)
    print(f'  {"Output":<6s}  {"1-step NRMS":>14s}  {f"{n_steps}-step NRMS":>14s}')
    print(f'  {"-"*6}  {"-"*14}  {"-"*14}')
    for i, name in enumerate(STAGE_NAMES):
        print(f'  {name:<6s}  {nrms_out_before[0, i]:>14.4e}  {nrms_out_before[-1, i]:>14.4e}')

    # Analytical n-step: feed analytical states through RK4
    # Build analytical encoder output for the IO windows
    # (analytical states at the same time indices as IO windows)
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    M_io = len(uh_val)

    x_ana_full_norm = compute_analytical_baseline(val_y, norm)
    # x_ana_full_norm has M_state = N - history samples, starting at index 'history'
    # IO windows start at the same 'history' index but have M_io = N - history - n_steps samples
    x_ana_io = x_ana_full_norm[:M_io]  # (M_io, 6), same time indices as IO windows

    # Roll analytical states through state block for n-step prediction
    state_block.eval()
    with torch.no_grad():
        x_ana_t = torch.tensor(x_ana_io, dtype=DTYPE_PT).unsqueeze(-1)  # (M_io, 6, 1)
        uf_val_t = torch.tensor(uf_val, dtype=DTYPE_PT)
        y_hat_ana_list = []
        x_roll = x_ana_t
        for step in range(n_steps):
            u_step = uf_val_t[:, step, :].unsqueeze(-1)
            z = torch.cat([x_roll, u_step], dim=1)
            x_roll = state_block.nonlinear_function(z)
            x_phys = x_roll * std_x_t + x_mean_t
            y_hat_step = (Cd_t @ x_phys).squeeze(-1)
            y_hat_ana_list.append(y_hat_step)
        y_hat_ana_steps = torch.stack(y_hat_ana_list, dim=1).numpy()

    nrms_out_ana = compute_output_nrms_per_horizon(y_hat_ana_steps, yf_val)
    print(f'\n  Analytical {n_steps}-step output prediction:')
    print(f'  {"Output":<6s}  {"1-step NRMS":>14s}  {f"{n_steps}-step NRMS":>14s}')
    print(f'  {"-"*6}  {"-"*14}  {"-"*14}')
    for i, name in enumerate(STAGE_NAMES):
        print(f'  {name:<6s}  {nrms_out_ana[0, i]:>14.4e}  {nrms_out_ana[-1, i]:>14.4e}')

    # =================================================================
    # TRAINING A: State MSE loss (Hoekstra 2026, Eq. 35)
    # =================================================================
    print(f'\n{"="*70}')
    print(f'TRAINING A: State MSE ({HP["epochs"]} epochs, lr={HP["lr"]}, '
          f'batch={HP["batch_size"]})')
    print('='*70)
    optimizer_smse = torch.optim.Adam(encoder_smse.parameters(), lr=HP['lr'])
    scheduler_smse = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_smse, mode='min',
        factor=HP['scheduler_factor'],
        patience=HP['scheduler_patience'],
        verbose=False,
    )

    u_train_state_t = torch.tensor(u_train_state, dtype=DTYPE_PT)
    y_train_state_t = torch.tensor(y_train_state, dtype=DTYPE_PT)
    x_train_target_t = torch.tensor(x_train_target, dtype=DTYPE_PT)

    sx_uh_val_t = torch.tensor(sx_uh_val, dtype=DTYPE_PT)
    sx_yh_val_t = torch.tensor(sx_yh_val, dtype=DTYPE_PT)
    sx_xt_val_t = torch.tensor(sx_xt_val, dtype=DTYPE_PT)

    N_train_state = len(u_train_state_t)
    batch_size = HP['batch_size']
    smse_train_losses = []
    smse_val_losses = []

    best_smse_val_loss = float('inf')
    best_smse_state = copy.deepcopy(encoder_smse.state_dict())
    best_smse_epoch = -1

    t_start_smse = time.time()
    for epoch in range(HP['epochs']):
        encoder_smse.train()
        perm = torch.randperm(N_train_state)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_train_state, batch_size):
            idx = perm[start: start + batch_size]
            x_hat = encoder_smse(u_train_state_t[idx], y_train_state_t[idx])
            loss = torch.mean((x_hat - x_train_target_t[idx]) ** 2)

            optimizer_smse.zero_grad()
            loss.backward()
            optimizer_smse.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        smse_train_losses.append(avg_train_loss)

        # Validation loss (state MSE)
        encoder_smse.eval()
        with torch.no_grad():
            x_hat_val = encoder_smse(sx_uh_val_t, sx_yh_val_t)
            val_loss = torch.mean((x_hat_val - sx_xt_val_t) ** 2).item()
        smse_val_losses.append(val_loss)

        scheduler_smse.step(val_loss)

        if val_loss < best_smse_val_loss:
            best_smse_val_loss = val_loss
            best_smse_state = copy.deepcopy(encoder_smse.state_dict())
            best_smse_epoch = epoch + 1

        current_lr = optimizer_smse.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t_start_smse
            best_flag = ' *' if best_smse_epoch == epoch + 1 else ''
            print(f'  Epoch {epoch+1:4d}/{HP["epochs"]}  '
                  f'train={avg_train_loss:.4e}  val={val_loss:.4e}  '
                  f'lr={current_lr:.1e}  [best@{best_smse_epoch}]  '
                  f'[{elapsed:.0f}s]{best_flag}')

    elapsed_smse = time.time() - t_start_smse
    print(f'\nState MSE training complete in {elapsed_smse:.0f}s')
    print(f'  Best val loss: {best_smse_val_loss:.4e} at epoch {best_smse_epoch}')
    encoder_smse.load_state_dict(best_smse_state)
    print(f'  Loaded best encoder (epoch {best_smse_epoch})')

    # =================================================================
    # TRAINING B: Output prediction loss (n-step rollout)
    # =================================================================
    print(f'\n{"="*70}')
    print(f'TRAINING B: Output prediction ({HP["epochs"]} epochs, lr={HP["lr"]}, '
          f'batch={HP["batch_size"]}, n_steps={n_steps}, '
          f'up_sample={HP["up_sample"]})')
    print('='*70)
    encoder.train()
    state_block.eval()  # dynamics block is frozen (no trainable params)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=HP['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=HP['scheduler_factor'],
        patience=HP['scheduler_patience'],
        verbose=False,
    )

    u_train_t = torch.tensor(u_train, dtype=DTYPE_PT)
    y_train_t = torch.tensor(y_train, dtype=DTYPE_PT)
    uf_train_t = torch.tensor(uf_train, dtype=DTYPE_PT)
    yf_train_t = torch.tensor(yf_train, dtype=DTYPE_PT)

    uh_val_t = torch.tensor(uh_val, dtype=DTYPE_PT)
    yh_val_t = torch.tensor(yh_val, dtype=DTYPE_PT)
    uf_val_t_full = torch.tensor(uf_val, dtype=DTYPE_PT)
    yf_val_t = torch.tensor(yf_val, dtype=DTYPE_PT)

    N_train = len(u_train_t)
    train_losses = []
    val_losses = []

    # Best-model checkpointing
    best_val_loss = float('inf')
    best_encoder_state = copy.deepcopy(encoder.state_dict())
    best_epoch = -1

    t_start = time.time()
    for epoch in range(HP['epochs']):
        encoder.train()
        perm = torch.randperm(N_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_train, batch_size):
            idx = perm[start: start + batch_size]

            y_hat_steps = nstep_rollout(
                encoder, state_block,
                u_train_t[idx], y_train_t[idx], uf_train_t[idx],
                std_x_t, x_mean_t, Cd_t,
            )
            loss = output_prediction_loss(y_hat_steps, yf_train_t[idx], ystd_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        # Validation loss
        encoder.eval()
        with torch.no_grad():
            y_hat_val_steps = nstep_rollout(
                encoder, state_block,
                uh_val_t, yh_val_t, uf_val_t_full,
                std_x_t, x_mean_t, Cd_t,
            )
            val_loss = output_prediction_loss(
                y_hat_val_steps, yf_val_t, ystd_t).item()
        val_losses.append(val_loss)

        # LR scheduler step
        scheduler.step(val_loss)

        # Best-model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = copy.deepcopy(encoder.state_dict())
            best_epoch = epoch + 1

        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            best_flag = ' *' if best_epoch == epoch + 1 else ''
            print(f'  Epoch {epoch+1:4d}/{HP["epochs"]}  '
                  f'train={avg_train_loss:.4e}  val={val_loss:.4e}  '
                  f'lr={current_lr:.1e}  [best@{best_epoch}]  '
                  f'[{elapsed:.0f}s]{best_flag}')

    elapsed_total = time.time() - t_start
    print(f'\nTraining complete in {elapsed_total:.0f}s')
    print(f'  Best validation loss: {best_val_loss:.4e} at epoch {best_epoch}')

    # Load best model
    encoder.load_state_dict(best_encoder_state)
    print(f'  Loaded best encoder (epoch {best_epoch})')

    # =================================================================
    # TRAINING C: State MSE with baseline-simulated states
    # =================================================================
    print(f'\n{"="*70}')
    print(f'TRAINING C: Baseline-simulated state MSE ({HP["epochs"]} epochs, '
          f'lr={HP["lr"]}, batch={HP["batch_size"]})')
    print('='*70)
    optimizer_bsim = torch.optim.Adam(encoder_bsim.parameters(), lr=HP['lr'])
    scheduler_bsim = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_bsim, mode='min',
        factor=HP['scheduler_factor'],
        patience=HP['scheduler_patience'],
        verbose=False,
    )

    u_train_bsim_t = torch.tensor(u_train_bsim, dtype=DTYPE_PT)
    y_train_bsim_t = torch.tensor(y_train_bsim, dtype=DTYPE_PT)
    x_train_bsim_t = torch.tensor(x_train_bsim, dtype=DTYPE_PT)

    bsim_uh_val_t = torch.tensor(bsim_uh_val, dtype=DTYPE_PT)
    bsim_yh_val_t = torch.tensor(bsim_yh_val, dtype=DTYPE_PT)
    bsim_xt_val_t = torch.tensor(bsim_xt_val, dtype=DTYPE_PT)

    N_train_bsim = len(u_train_bsim_t)
    bsim_train_losses = []
    bsim_val_losses = []

    best_bsim_val_loss = float('inf')
    best_bsim_state = copy.deepcopy(encoder_bsim.state_dict())
    best_bsim_epoch = -1

    t_start_bsim = time.time()
    for epoch in range(HP['epochs']):
        encoder_bsim.train()
        perm = torch.randperm(N_train_bsim)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_train_bsim, batch_size):
            idx = perm[start: start + batch_size]
            x_hat = encoder_bsim(u_train_bsim_t[idx], y_train_bsim_t[idx])
            loss = torch.mean((x_hat - x_train_bsim_t[idx]) ** 2)

            optimizer_bsim.zero_grad()
            loss.backward()
            optimizer_bsim.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        bsim_train_losses.append(avg_train_loss)

        # Validation loss (against baseline-simulated targets)
        encoder_bsim.eval()
        with torch.no_grad():
            x_hat_val = encoder_bsim(bsim_uh_val_t, bsim_yh_val_t)
            val_loss = torch.mean((x_hat_val - bsim_xt_val_t) ** 2).item()
        bsim_val_losses.append(val_loss)

        scheduler_bsim.step(val_loss)

        if val_loss < best_bsim_val_loss:
            best_bsim_val_loss = val_loss
            best_bsim_state = copy.deepcopy(encoder_bsim.state_dict())
            best_bsim_epoch = epoch + 1

        current_lr = optimizer_bsim.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t_start_bsim
            best_flag = ' *' if best_bsim_epoch == epoch + 1 else ''
            print(f'  Epoch {epoch+1:4d}/{HP["epochs"]}  '
                  f'train={avg_train_loss:.4e}  val={val_loss:.4e}  '
                  f'lr={current_lr:.1e}  [best@{best_bsim_epoch}]  '
                  f'[{elapsed:.0f}s]{best_flag}')

    elapsed_bsim = time.time() - t_start_bsim
    print(f'\nBaseline-sim state MSE training complete in {elapsed_bsim:.0f}s')
    print(f'  Best val loss: {best_bsim_val_loss:.4e} at epoch {best_bsim_epoch}')
    encoder_bsim.load_state_dict(best_bsim_state)
    print(f'  Loaded best encoder (epoch {best_bsim_epoch})')

    # =================================================================
    # COMPARISON: Evaluate all three encoders
    # =================================================================
    print(f'\n{"="*70}')
    print('COMPARISON: State MSE vs Output Prediction vs Baseline-Sim')
    print('='*70)

    # --- State quality (all evaluated against TRUE states) ---
    x_hat_smse, nrms_smse = evaluate_encoder_states(
        encoder_smse, sx_uh_val, sx_yh_val, sx_xt_val)
    x_hat_opl, nrms_opl = evaluate_encoder_states(
        encoder, sx_uh_val, sx_yh_val, sx_xt_val)
    x_hat_bsim, nrms_bsim = evaluate_encoder_states(
        encoder_bsim, sx_uh_val, sx_yh_val, sx_xt_val)

    # --- N-step output prediction ---
    y_hat_smse, nrms_out_smse = evaluate_nstep(
        encoder_smse, state_block, uh_val, yh_val, uf_val, yf_val, norm)
    y_hat_opl, nrms_out_opl = evaluate_nstep(
        encoder, state_block, uh_val, yh_val, uf_val, yf_val, norm)
    y_hat_bsim, nrms_out_bsim = evaluate_nstep(
        encoder_bsim, state_block, uh_val, yh_val, uf_val, yf_val, norm)

    # --- Print state NRMS comparison ---
    print(f'\n--- State NRMS comparison (all vs true states) ---')
    print(f'  {"State":<8s}  {"Init":>12s}  {"TrueMSE":>12s}  '
          f'{"BsimMSE":>12s}  {"OutPred":>12s}  {"Analyt":>12s}')
    print(f'  {"-"*8}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {nrms_before[i]:>12.4e}  {nrms_smse[i]:>12.4e}  '
              f'{nrms_bsim[i]:>12.4e}  {nrms_opl[i]:>12.4e}  {nrms_ana[i]:>12.4e}')

    # Physical-unit RMS errors
    x_smse_phys = denormalize_states(x_hat_smse, norm)
    x_opl_phys = denormalize_states(x_hat_opl, norm)
    x_bsim_phys = denormalize_states(x_hat_bsim, norm)
    x_ana_phys = denormalize_states(x_hat_ana, norm)
    x_gt_phys = denormalize_states(sx_xt_val, norm)
    rms_smse = compute_rms_error(x_smse_phys, x_gt_phys)
    rms_opl = compute_rms_error(x_opl_phys, x_gt_phys)
    rms_bsim = compute_rms_error(x_bsim_phys, x_gt_phys)
    rms_ana = compute_rms_error(x_ana_phys, x_gt_phys)

    print(f'\n  Per-channel RMS error (logical coordinates):')
    print(f'  {"State":<8s}  {"TrueMSE":>12s}  {"BsimMSE":>12s}  '
          f'{"OutPred":>12s}  {"Analyt":>12s}  {"Unit":>6s}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {rms_smse[i]:>12.4e}  {rms_bsim[i]:>12.4e}  '
              f'{rms_opl[i]:>12.4e}  {rms_ana[i]:>12.4e}  {PHYS_UNITS[i]:>6s}')

    # --- Print output NRMS comparison ---
    print(f'\n--- {n_steps}-step output NRMS comparison ---')
    print(f'  {"Output":<6s}  {"Init":>12s}  {"TrueMSE":>12s}  '
          f'{"BsimMSE":>12s}  {"OutPred":>12s}  {"Analyt":>12s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}')
    for i, name in enumerate(STAGE_NAMES):
        print(f'  {name:<6s}  {nrms_out_before[0, i]:>12.4e}  '
              f'{nrms_out_smse[0, i]:>12.4e}  {nrms_out_bsim[0, i]:>12.4e}  '
              f'{nrms_out_opl[0, i]:>12.4e}  {nrms_out_ana[0, i]:>12.4e}')

    print(f'\n  {n_steps}-step:')
    for i, name in enumerate(STAGE_NAMES):
        print(f'  {name:<6s}  {nrms_out_before[-1, i]:>12.4e}  '
              f'{nrms_out_smse[-1, i]:>12.4e}  {nrms_out_bsim[-1, i]:>12.4e}  '
              f'{nrms_out_opl[-1, i]:>12.4e}  {nrms_out_ana[-1, i]:>12.4e}')

    # --- Verdict ---
    print(f'\n--- VERDICT ---')
    methods = {
        'Model-based init': (np.max(nrms_out_before[-1]), np.max(nrms_before)),
        'True state MSE':   (np.max(nrms_out_smse[-1]),   np.max(nrms_smse)),
        'Baseline-sim MSE': (np.max(nrms_out_bsim[-1]),   np.max(nrms_bsim)),
        'Output pred':      (np.max(nrms_out_opl[-1]),    np.max(nrms_opl)),
        'Analytical':       (np.max(nrms_out_ana[-1]),     np.max(nrms_ana)),
    }
    print(f'  {"Method":<20s}  {f"{n_steps}-step NRMS":>14s}  {"State NRMS":>12s}')
    print(f'  {"-"*20}  {"-"*14}  {"-"*12}')
    for name, (out_nrms, st_nrms) in methods.items():
        print(f'  {name:<20s}  {out_nrms:>14.4e}  {st_nrms:>12.4e}')

    # Find winners (among trained methods only)
    trained = {'True state MSE': methods['True state MSE'],
               'Baseline-sim MSE': methods['Baseline-sim MSE'],
               'Output pred': methods['Output pred']}
    best_output = min(trained, key=lambda k: trained[k][0])
    best_state = min(trained, key=lambda k: trained[k][1])
    print(f'\n  Best {n_steps}-step output: {best_output}')
    print(f'  Best state reconstruction: {best_state}')

    # =================================================================
    # Save outputs
    # =================================================================

    # --- Encoder weights ---
    torch.save(encoder.state_dict(),
               os.path.join(OUT_DIR, 'encoder_opl_weights.pt'))
    torch.save(encoder_smse.state_dict(),
               os.path.join(OUT_DIR, 'encoder_smse_weights.pt'))
    torch.save(encoder_bsim.state_dict(),
               os.path.join(OUT_DIR, 'encoder_bsim_weights.pt'))
    print(f'\nSaved encoder weights to {OUT_DIR}')

    # --- Results JSON ---
    results = dict(
        # Training A: true state MSE
        smse_state_nrms={name: float(nrms_smse[i])
                         for i, name in enumerate(STATE_NAMES)},
        smse_output_nrms_1step={name: float(nrms_out_smse[0, i])
                                for i, name in enumerate(STAGE_NAMES)},
        smse_output_nrms_nstep={name: float(nrms_out_smse[-1, i])
                                for i, name in enumerate(STAGE_NAMES)},
        smse_state_rms={name: float(rms_smse[i])
                        for i, name in enumerate(STATE_NAMES)},
        smse_train_time_s=elapsed_smse,
        smse_best_epoch=best_smse_epoch,
        smse_best_val_loss=best_smse_val_loss,
        # Training B: output prediction
        opl_state_nrms={name: float(nrms_opl[i])
                        for i, name in enumerate(STATE_NAMES)},
        opl_output_nrms_1step={name: float(nrms_out_opl[0, i])
                               for i, name in enumerate(STAGE_NAMES)},
        opl_output_nrms_nstep={name: float(nrms_out_opl[-1, i])
                               for i, name in enumerate(STAGE_NAMES)},
        opl_state_rms={name: float(rms_opl[i])
                       for i, name in enumerate(STATE_NAMES)},
        opl_train_time_s=elapsed_total,
        opl_best_epoch=best_epoch,
        opl_best_val_loss=best_val_loss,
        # Training C: baseline-simulated state MSE
        bsim_state_nrms={name: float(nrms_bsim[i])
                         for i, name in enumerate(STATE_NAMES)},
        bsim_output_nrms_1step={name: float(nrms_out_bsim[0, i])
                                for i, name in enumerate(STAGE_NAMES)},
        bsim_output_nrms_nstep={name: float(nrms_out_bsim[-1, i])
                                for i, name in enumerate(STAGE_NAMES)},
        bsim_state_rms={name: float(rms_bsim[i])
                        for i, name in enumerate(STATE_NAMES)},
        bsim_train_time_s=elapsed_bsim,
        bsim_best_epoch=best_bsim_epoch,
        bsim_best_val_loss=best_bsim_val_loss,
        bsim_vs_true_nrms={name: float(bsim_vs_true_nrms[i])
                           for i, name in enumerate(STATE_NAMES)},
        # Baselines
        init_state_nrms={name: float(nrms_before[i])
                         for i, name in enumerate(STATE_NAMES)},
        analytical_state_nrms={name: float(nrms_ana[i])
                               for i, name in enumerate(STATE_NAMES)},
        analytical_output_nrms_1step={name: float(nrms_out_ana[0, i])
                                      for i, name in enumerate(STAGE_NAMES)},
        analytical_output_nrms_nstep={name: float(nrms_out_ana[-1, i])
                                      for i, name in enumerate(STAGE_NAMES)},
        vel_verification_nrms={name: float(vel_nrms[i])
                               for i, name in enumerate(STATE_NAMES[3:])},
        n_params=n_params,
        n_transient=N_TRANSIENT,
        hp=HP,
    )
    json_path = os.path.join(OUT_DIR, 'encoder_io_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {json_path}')

    # --- Trajectory data ---
    npz_path = os.path.join(OUT_DIR, 'encoder_io_data.npz')
    np.savez_compressed(npz_path,
        # Training A: true state MSE
        smse_x_norm=x_hat_smse,
        smse_y_hat_steps=y_hat_smse,
        smse_output_nrms_per_horizon=nrms_out_smse,
        smse_train_losses=smse_train_losses,
        smse_val_losses=smse_val_losses,
        # Training B: output prediction
        opl_x_norm=x_hat_opl,
        opl_y_hat_steps=y_hat_opl,
        opl_output_nrms_per_horizon=nrms_out_opl,
        opl_train_losses=train_losses,
        opl_val_losses=val_losses,
        # Training C: baseline-simulated state MSE
        bsim_x_norm=x_hat_bsim,
        bsim_y_hat_steps=y_hat_bsim,
        bsim_output_nrms_per_horizon=nrms_out_bsim,
        bsim_train_losses=bsim_train_losses,
        bsim_val_losses=bsim_val_losses,
        # Baselines
        x_before_norm=x_hat_before,
        x_analytical_norm=x_hat_ana,
        x_target_norm=sx_xt_val,
        y_target=yf_val,
        y_hat_ana_steps=y_hat_ana_steps,
        output_nrms_per_horizon_init=nrms_out_before,
        output_nrms_per_horizon_ana=nrms_out_ana,
        # Normalization
        std_x=norm['std_x'], x_mean=norm['x_mean'], ystd=norm['ystd'],
        # Metadata
        state_names=STATE_NAMES, stage_names=STAGE_NAMES,
        fs=FS_NEW, n_steps=n_steps,
    )
    print(f'Saved: {npz_path}')

    # =================================================================
    # Plots
    # =================================================================

    # 1. Loss curves (three methods)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.semilogy(smse_train_losses, label='train', linewidth=0.8)
    ax1.semilogy(smse_val_losses, label='val', linewidth=0.8)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('State MSE')
    ax1.set_title('A: True state MSE')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.semilogy(train_losses, label='train', linewidth=0.8)
    ax2.semilogy(val_losses, label='val', linewidth=0.8)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Output prediction MSE')
    ax2.set_title(f'B: Output prediction ({n_steps}-step)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3.semilogy(bsim_train_losses, label='train', linewidth=0.8)
    ax3.semilogy(bsim_val_losses, label='val', linewidth=0.8)
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('State MSE')
    ax3.set_title('C: Baseline-sim state MSE')
    ax3.legend(); ax3.grid(True, alpha=0.3)
    fig.suptitle('Encoder training loss curves', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'encoder_comparison_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: encoder_comparison_loss.png')

    # 2. State NRMS bar chart (5 methods)
    n = len(STATE_NAMES)
    x_pos = np.arange(n)
    w = 0.15
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x_pos - 2*w, nrms_before, w, label='model-based init',
           color='tab:orange', alpha=0.8)
    ax.bar(x_pos - w, nrms_smse, w, label='A: true state MSE',
           color='tab:green', alpha=0.8)
    ax.bar(x_pos, nrms_bsim, w, label='C: baseline-sim MSE',
           color='tab:purple', alpha=0.8)
    ax.bar(x_pos + w, nrms_opl, w, label='B: output pred',
           color='tab:red', alpha=0.8)
    ax.bar(x_pos + 2*w, nrms_ana, w, label='analytical baseline',
           color='tab:blue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('State NRMS comparison (all vs true states)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'encoder_comparison_state_nrms.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: encoder_comparison_state_nrms.png')

    # 3. NRMS vs prediction horizon (5 methods)
    horizons = np.arange(1, n_steps + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.semilogy(horizons, nrms_out_before[:, i], '--', color='tab:orange',
                    markersize=3, linewidth=1.0, label='model-based init')
        ax.semilogy(horizons, nrms_out_smse[:, i], '-o', color='tab:green',
                    markersize=3, linewidth=1.0, label='A: true state MSE')
        ax.semilogy(horizons, nrms_out_bsim[:, i], '-d', color='tab:purple',
                    markersize=3, linewidth=1.0, label='C: baseline-sim MSE')
        ax.semilogy(horizons, nrms_out_opl[:, i], '-s', color='tab:red',
                    markersize=3, linewidth=1.0, label='B: output pred')
        ax.semilogy(horizons, nrms_out_ana[:, i], '-^', color='tab:blue',
                    markersize=3, linewidth=1.0, label='analytical')
        ax.set_ylabel(f'{STAGE_NAMES[i]} NRMS')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Prediction horizon [steps]')
    fig.suptitle('Output NRMS vs horizon', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'encoder_comparison_horizon_nrms.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: encoder_comparison_horizon_nrms.png')

    # 4. State trajectories (all encoders vs target)
    T = min(2000, len(x_hat_smse))
    t = np.arange(T) / FS_NEW
    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(14, 2.5 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_gt_phys[:T, i], 'k-', linewidth=0.8, label='target')
        ax.plot(t, x_smse_phys[:T, i], '-', color='tab:green', linewidth=0.8,
                label=f'A: true MSE (NRMS={nrms_smse[i]:.2e})')
        ax.plot(t, x_bsim_phys[:T, i], '-', color='tab:purple', linewidth=0.8,
                label=f'C: bsim MSE (NRMS={nrms_bsim[i]:.2e})')
        ax.plot(t, x_opl_phys[:T, i], '--', color='tab:red', linewidth=0.8,
                label=f'B: out pred (NRMS={nrms_opl[i]:.2e})')
        ax.plot(t, x_ana_phys[:T, i], ':', color='tab:blue', linewidth=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e})')
        ax.set_ylabel(f'{STATE_NAMES[i]} [{PHYS_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('State reconstruction comparison', y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'encoder_comparison_states.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: encoder_comparison_states.png')

    print('\n' + '=' * 70)
    print('Encoder I/O validation complete.')


if __name__ == '__main__':
    main()
