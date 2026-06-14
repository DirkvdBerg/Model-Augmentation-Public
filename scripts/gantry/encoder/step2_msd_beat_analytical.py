"""
step2_msd_beat_analytical.py
----------------------------
Encoder verification Step 2: MSD system, encoder must beat analytical baseline.

Data is generated with USE_MSD=true in MATLAB (hidden mass-spring-damper on
the Y-axis payload). The baseline FP model has no MSD knowledge, so:
  - Analytical baseline (P_inv + FD) cannot account for MSD effects
  - Trained encoder uses full (u_hist, y_hist) window and can learn MSD-related
    state corrections from I/O patterns

Expected: trained encoder beats analytical mainly on velocity channels and Y-axis.

Both are compared against x_logical from .mat files (6 physical states of the
gantry, excluding the MSD state). Optional Step 2b: correlate augmented states
with delta_a (MSD relative displacement).

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/step2_msd_beat_analytical.py
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

# --- Project root ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.utils.utils import (
    normalize_linear_ss_matrices, selection_matrix, expansion_matrix,
)
from model_augmentation.utils.torch_nets import (
    zero_init_feed_forward_nn, LinearInitEncoderWrapper,
)
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None  # LPV self-scheduled

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
na_right = 1  # required by linear_encoder_init (reconstructability needs y(k))
nb_right = 1

# Training hyperparameters (justified from Jan's meeting advice)
HP = dict(
    NX_ANN=2,              # augmented states to capture MSD dynamics
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    nf=20,                 # HEURISTIC: Jan: "problem might be really easy", start small
    batch_size=128,        # HEURISTIC: Jan: "take less number of batch sizes"
    lr=1e-5,               # HEURISTIC: Jan: "train with low learning rate"
    epochs=50,
)

# Data: MSD directory (USE_MSD=true, MA_FRAC=0.10 -> root multisine/)
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

# Output
OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
PHYS_UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s']


def compute_rms_error(x_hat, x_target):
    """Per-channel RMS error (same units as input)."""
    return np.sqrt(np.mean((x_hat - x_target)**2, axis=0))

# =============================================================================
# Data loading
# =============================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_mat(filename):
    """Load u, y, x_logical, delta_a from .mat file, downsample to FS_NEW."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP) if 'u_total' in d else d['u'][::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    delta_a = d['delta_a'][::D].astype(DTYPE_NP) if 'delta_a' in d else None
    return u, y, x_logical, delta_a


# =============================================================================
# Normalization (from x_logical, same as step0/step1)
# =============================================================================

def compute_normalization(train_data):
    """Compute normalization constants from training data using x_logical."""
    u_all = np.concatenate([u for u, _, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _, _ in train_data])
    x_all = np.concatenate([x for _, _, x, _ in train_data])

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0 = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    return dict(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )


# =============================================================================
# Analytical baseline (P_inv + finite-diff)
# =============================================================================

def compute_analytical_baseline(y, x_logical):
    """Analytical state estimates: P_inv for positions, backward FD for velocities.

    Returns per-channel NRMS against x_logical (un-normalized, physical units).
    """
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    # THEORY: q_logical = P_inv @ y_stage (measurement equation)
    pos = (P_inv_T @ y.T).T  # (N, 3)

    # HEURISTIC: backward finite difference for velocity, O(Ts) accurate
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0] = vel[1]

    x_analytical = np.hstack([pos, vel])  # (N, 6)

    # Per-channel NRMS vs x_logical
    err = x_analytical - x_logical
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_gt = np.sqrt(np.mean(x_logical**2, axis=0))
    nrms = rms_err / (rms_gt + 1e-12)

    return nrms, x_analytical


# =============================================================================
# Build model
# =============================================================================

def build_model(hp, norm):
    """Build interconnect + SSE_Interconnect with linear_encoder_init."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    x_mean = norm['x_mean']
    std_x = norm['std_x']
    std_u = norm['std_u']
    u_mean = norm['u_mean']
    ystd = norm['ystd']
    y0 = norm['y0']

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np = Dd.numpy()

    PHY_IX = np.arange(NX_PHYS)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=hp['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_block)

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    # Manual normalization
    fit_sys.norm.u0 = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0 = y0
    fit_sys.norm.ystd = ystd

    # --- Encoder: linear_encoder_init (Hoekstra 2026 Eq. 16-17) ---
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

    fit_sys.encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder,
        nx_ann=NX_ANN,
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    return fit_sys


# =============================================================================
# Evaluate encoder states via saved_output_signals
# =============================================================================

def evaluate_encoder_states(fit_sys, val_sys_data, x_logical_val, norm):
    """Run apply_experiment, extract evolved states, compare to x_logical.

    Returns per-channel NRMS (physical states only), plus augmented states.
    """
    NX_ANN = fit_sys.hfn.nx - NX_PHYS
    x_mean = norm['x_mean']
    std_x = norm['std_x']

    fit_sys.eval()
    fit_sys.hfn.reset_saved_signals()
    sim_result = fit_sys.apply_experiment(val_sys_data)
    cheat_n = sim_result.cheat_n

    # saved_output_signals: (total_dim, T), first nxd rows = state
    saved = np.array(fit_sys.hfn.saved_output_signals)
    x_enc_norm = saved[:NX_PHYS, :]  # (6, T) physical states (normalized)
    x_enc_phys = (x_enc_norm * std_x + x_mean).T  # (T, 6)

    x_aug_norm = None
    if NX_ANN > 0:
        x_aug_norm = saved[NX_PHYS:NX_PHYS + NX_ANN, :].T  # (T, NX_ANN)

    # Align with ground truth (skip cheat_n initial samples)
    x_gt = x_logical_val[cheat_n:]
    T = min(len(x_enc_phys), len(x_gt))
    x_enc_phys = x_enc_phys[:T]
    x_gt = x_gt[:T]
    if x_aug_norm is not None:
        x_aug_norm = x_aug_norm[:T]

    # Per-channel NRMS
    err = x_enc_phys - x_gt
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_gt = np.sqrt(np.mean(x_gt**2, axis=0))
    nrms = rms_err / (rms_gt + 1e-12)

    return nrms, x_enc_phys, x_gt, x_aug_norm, cheat_n


# =============================================================================
# Plotting
# =============================================================================

def plot_state_comparison(x_enc, x_ana, x_gt, nrms_enc, nrms_ana, cheat_n,
                          title, out_path):
    """Time-domain: encoder vs analytical vs ground truth."""
    T = min(2000, len(x_enc))
    t = np.arange(T) / FS_NEW + cheat_n / FS_NEW

    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(14, 2.5 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_gt[:T, i], 'k-', linewidth=0.8, label='x_logical (ground truth)')
        ax.plot(t, x_enc[:T, i], 'r--', linewidth=0.8,
                label=f'encoder (NRMS={nrms_enc[i]:.2e})')
        ax.plot(t, x_ana[:T, i], 'b:', linewidth=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e})')
        ax.set_ylabel(STATE_NAMES[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_state_error(x_enc, x_ana, x_gt, cheat_n, title, out_path):
    """Error time series: encoder error vs analytical error."""
    T = min(2000, len(x_enc))
    t = np.arange(T) / FS_NEW + cheat_n / FS_NEW

    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(14, 2.5 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        err_enc = x_enc[:T, i] - x_gt[:T, i]
        err_ana = x_ana[:T, i] - x_gt[:T, i]
        ax.plot(t, err_enc, 'r-', linewidth=0.6, label='encoder error')
        ax.plot(t, err_ana, 'b-', linewidth=0.6, label='analytical error')
        ax.set_ylabel(f'{STATE_NAMES[i]} [{PHYS_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_nrms_bar(nrms_dict, title, out_path):
    """Bar chart: NRMS per channel for multiple methods."""
    x = np.arange(NX_PHYS)
    n = len(nrms_dict)
    w = 0.8 / n
    colors = ['tab:orange', 'tab:red', 'tab:blue']
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (label, nrms) in enumerate(nrms_dict.items()):
        ax.bar(x + (j - n/2 + 0.5) * w, nrms, w, label=label, color=colors[j % len(colors)], alpha=0.8)
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


def plot_augmented_vs_delta_a(x_aug, delta_a, cheat_n, out_path):
    """Step 2b: correlate augmented states with MSD displacement delta_a.

    Informational, not a pass/fail criterion.
    """
    if x_aug is None or delta_a is None:
        print('Skipping augmented state plot (no augmented states or no delta_a)')
        return

    T = min(len(x_aug), len(delta_a))
    t = np.arange(T) / FS_NEW + cheat_n / FS_NEW
    n_aug = x_aug.shape[1]

    fig, axes = plt.subplots(n_aug + 1, 1, figsize=(14, 3 * (n_aug + 1)), sharex=True)
    if n_aug + 1 == 1:
        axes = [axes]

    # delta_a
    axes[0].plot(t, delta_a[:T], 'k-', linewidth=0.8)
    axes[0].set_ylabel('delta_a [m]')
    axes[0].set_title('MSD relative displacement (ground truth)')
    axes[0].grid(True, alpha=0.3)

    # Augmented states (normalized, no physical interpretation guaranteed)
    for j in range(n_aug):
        ax = axes[j + 1]
        ax.plot(t, x_aug[:T, j], 'g-', linewidth=0.8)
        # Compute correlation with delta_a
        corr = np.corrcoef(x_aug[:T, j], delta_a[:T])[0, 1]
        ax.set_ylabel(f'x_aug[{j}]')
        ax.set_title(f'Augmented state {j} (corr with delta_a: {corr:.3f})')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Step 2b: Augmented states vs MSD displacement', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_loss_curves(fit_sys, out_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if hasattr(fit_sys, 'Loss_train') and fit_sys.Loss_train:
        ax.plot(fit_sys.Loss_train, label='train')
    if hasattr(fit_sys, 'Loss_val') and fit_sys.Loss_val:
        ax.plot(fit_sys.Loss_val, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Step 2: Training loss (MSD data)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Step 2: MSD system, encoder must beat analytical baseline')
    print('=' * 70)
    print(f'HP: lr={HP["lr"]}, nf={HP["nf"]}, batch_size={HP["batch_size"]}, '
          f'epochs={HP["epochs"]}, NX_ANN={HP["NX_ANN"]}')
    print(f'Data: {TRAJ_DIR}')

    # --- Load data ---
    print(f'\nLoading MSD data...')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical, val_delta_a = load_mat(VAL_FILE)

    for i, (fname, (u, y, x, da)) in enumerate(zip(TRAIN_FILES, train_data)):
        print(f'  T{i+1} ({fname}): u={u.shape}, y={y.shape}, x={x.shape}, '
              f'delta_a={"present" if da is not None else "MISSING"}')
    print(f'  Val ({VAL_FILE}): u={val_u.shape}, y={val_y.shape}, '
          f'x={val_x_logical.shape}, delta_a={"present" if val_delta_a is not None else "MISSING"}')

    # --- Normalization ---
    norm = compute_normalization(train_data)
    print(f'\nNormalization:')
    print(f'  std_x = {norm["std_x"].flatten()}')
    print(f'  std_u = {norm["std_u"].flatten()}')

    # --- Analytical baseline ---
    print('\n--- Analytical baseline (P_inv + FD) on validation ---')
    nrms_ana, x_analytical = compute_analytical_baseline(val_y, val_x_logical)
    rms_ana_phys = compute_rms_error(x_analytical, val_x_logical)
    print(f'  {"State":<6s}  {"NRMS":>12s}  {"RMS err":>12s}  {"Unit":<5s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*5}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<6s}  {nrms_ana[i]:>12.4e}  {rms_ana_phys[i]:>12.4e}  {PHYS_UNITS[i]:<5s}')

    # --- Build deepSI System_data ---
    train_list = [
        deepSI.System_data(u=u, y=y, dt=TS_NEW)
        for u, y, _, _ in train_data
    ]
    train_sys_data = deepSI.System_data_list(train_list)
    val_sys_data = deepSI.System_data(u=val_u, y=val_y, dt=TS_NEW)

    # --- Build model ---
    print('\nBuilding model...')
    fit_sys = build_model(HP, norm)
    fit_sys.init_model(sys_data=train_sys_data, auto_fit_norm=False)
    fit_sys.hfn.to(DTYPE_PT)

    n_params = 0
    for item in fit_sys.parameters_with_names.values():
        params = item['params']
        if isinstance(params, torch.nn.Parameter):
            n_params += params.numel()
        else:
            n_params += sum(p.numel() for p in params)
    print(f'Total trainable parameters: {n_params}')

    # --- Evaluate BEFORE training ---
    print('\n--- Encoder quality BEFORE training ---')
    nrms_init, x_enc_init, x_gt_init, x_aug_init, cheat_n = evaluate_encoder_states(
        fit_sys, val_sys_data, val_x_logical, norm)
    rms_init_phys = compute_rms_error(x_enc_init, x_gt_init)
    print(f'  {"State":<6s}  {"NRMS":>12s}  {"RMS err":>12s}  {"Unit":<5s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*5}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<6s}  {nrms_init[i]:>12.4e}  {rms_init_phys[i]:>12.4e}  {PHYS_UNITS[i]:<5s}')

    # --- Train ---
    list_val_measures = [f'{i}-step-RMS' for i in [1, 5, 20]]
    val_measure = f'{HP["nf"]}-step-RMS'

    print(f'\nTraining: {HP["epochs"]} epochs, validation_measure={val_measure}')
    print(f'Additional measures: {list_val_measures}')

    fit_sys.fit(
        train_sys_data=train_sys_data,
        val_sys_data=val_sys_data,
        batch_size=HP['batch_size'],
        epochs=HP['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': HP['nf']},
        optimizer_kwargs={'lr': HP['lr']},
        validation_measure=val_measure,
        list_val_measures=list_val_measures,
    )

    # --- Evaluate AFTER training (best checkpoint) ---
    fit_sys.checkpoint_load_system(name='_best')
    print('\n--- Encoder quality AFTER training (best checkpoint) ---')
    nrms_post, x_enc_post, x_gt_post, x_aug_post, cheat_n = evaluate_encoder_states(
        fit_sys, val_sys_data, val_x_logical, norm)
    rms_post_phys = compute_rms_error(x_enc_post, x_gt_post)
    print(f'  {"State":<6s}  {"NRMS":>12s}  {"RMS err":>12s}  {"Unit":<5s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*5}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<6s}  {nrms_post[i]:>12.4e}  {rms_post_phys[i]:>12.4e}  {PHYS_UNITS[i]:<5s}')

    # --- Pass/fail: encoder must beat analytical ---
    print('\n--- Comparison: encoder vs analytical ---')
    print(f'  {"State":<6s}  {"Encoder":>12s}  {"Analytical":>12s}  {"Winner":>10s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*10}')
    enc_wins = 0
    for i, name in enumerate(STATE_NAMES):
        winner = 'encoder' if nrms_post[i] < nrms_ana[i] else 'analytical'
        if nrms_post[i] < nrms_ana[i]:
            enc_wins += 1
        print(f'  {name:<6s}  {nrms_post[i]:>12.4e}  {nrms_ana[i]:>12.4e}  {winner:>10s}')

    print(f'\nEncoder wins {enc_wins}/{NX_PHYS} channels')
    if enc_wins == NX_PHYS:
        print('PASS: encoder beats analytical on ALL channels')
    elif enc_wins >= 4:
        print('PARTIAL PASS: encoder beats analytical on most channels')
    else:
        print('FAIL: encoder does not beat analytical on enough channels')

    # --- Save results ---
    results = dict(
        hp=HP,
        cheat_n=int(cheat_n),
        n_params=n_params,
        nrms_init={name: float(nrms_init[i]) for i, name in enumerate(STATE_NAMES)},
        nrms_post={name: float(nrms_post[i]) for i, name in enumerate(STATE_NAMES)},
        nrms_analytical={name: float(nrms_ana[i]) for i, name in enumerate(STATE_NAMES)},
        encoder_wins=enc_wins,
        loss_train=getattr(fit_sys, 'Loss_train', []),
        loss_val=getattr(fit_sys, 'Loss_val', []),
    )
    json_path = os.path.join(OUT_DIR, 'step2_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {json_path}')

    # Align analytical baseline to same time window as encoder (skip cheat_n)
    x_ana_aligned = x_analytical[cheat_n:]
    T = min(len(x_enc_post), len(x_ana_aligned))
    x_ana_aligned = x_ana_aligned[:T]

    # Save trajectories for plot reconstruction
    npz_data = dict(
        x_enc_init=x_enc_init, x_enc_post=x_enc_post,
        x_analytical=x_ana_aligned, x_gt=x_gt_post,
        nrms_init=nrms_init, nrms_post=nrms_post, nrms_analytical=nrms_ana,
        cheat_n=cheat_n, state_names=STATE_NAMES, fs=FS_NEW,
    )
    if x_aug_post is not None:
        npz_data['x_aug_post'] = x_aug_post
    da_aligned = val_delta_a[cheat_n:cheat_n + T] if val_delta_a is not None else None
    if da_aligned is not None:
        npz_data['delta_a'] = da_aligned
    npz_path = os.path.join(OUT_DIR, 'step2_data.npz')
    np.savez_compressed(npz_path, **npz_data)
    print(f'Saved: {npz_path}')

    # --- Plots ---
    plot_state_comparison(
        x_enc_init, x_ana_aligned, x_gt_init, nrms_init, nrms_ana, cheat_n,
        'Step 2: BEFORE training (MSD data)',
        os.path.join(OUT_DIR, 'step2_before_training.png'))
    plot_state_error(
        x_enc_init, x_ana_aligned, x_gt_init, cheat_n,
        'Step 2: BEFORE training error (physical units)',
        os.path.join(OUT_DIR, 'step2_before_error.png'))

    plot_state_comparison(
        x_enc_post, x_ana_aligned, x_gt_post, nrms_post, nrms_ana, cheat_n,
        'Step 2: AFTER training (MSD data)',
        os.path.join(OUT_DIR, 'step2_after_training.png'))
    plot_state_error(
        x_enc_post, x_ana_aligned, x_gt_post, cheat_n,
        'Step 2: AFTER training error (physical units)',
        os.path.join(OUT_DIR, 'step2_after_error.png'))

    plot_nrms_bar(
        {'init': nrms_init, 'after training': nrms_post, 'analytical': nrms_ana},
        'Step 2: NRMS per channel (before/after/analytical)',
        os.path.join(OUT_DIR, 'step2_nrms_bar.png'))

    plot_loss_curves(fit_sys, os.path.join(OUT_DIR, 'step2_loss.png'))

    # --- Step 2b: augmented states vs delta_a ---
    plot_augmented_vs_delta_a(
        x_aug_post, da_aligned, cheat_n,
        os.path.join(OUT_DIR, 'step2b_augmented_vs_delta_a.png'))


if __name__ == '__main__':
    main()
