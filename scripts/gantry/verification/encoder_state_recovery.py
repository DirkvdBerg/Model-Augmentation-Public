"""
encoder_state_recovery.py
-------------------------
Tests whether the default (learned) encoder can recover the 6 physical
states from input/output history alone, using physics-only dynamics.

Runs two sequential experiments:
  1. Multisine data (broadband excitation)
  2. Trajectory data (point-to-point motions)

For each experiment:
  - Builds a physics-only interconnect (no ANN) with default encoder
  - Trains with output prediction loss only
  - Compares encoder states to x_logical ground truth from MATLAB
  - Reports per-state MAE, correlation, scale ratio, and sim-NRMS

Plots (per experiment):
  1. Loss convergence (train + val)
  2. Per-state MAE convergence over training checkpoints
  3. Scatter: encoder state vs analytical state per physical state
  4. 6x6 correlation matrix heatmap

Usage:
  conda run -n GraduationProject python scripts/gantry/verification/encoder_state_recovery.py
  python encoder_state_recovery.py --epochs 100
  python encoder_state_recovery.py --mode multisine   # multisine only
  python encoder_state_recovery.py --mode trajectories # trajectories only
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.gantry_ss import Cd, Dd, P


# =========================================================================
# Configuration
# =========================================================================

NX_PHYS = 6
nu, ny  = 3, 3
Y_OP    = None
SEED    = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

NF_SECONDS   = 0.300   # [s] rollout horizon for training loss
NANB_SECONDS = 0.030   # [s] encoder history window

DEFAULT_HP = dict(
    n_nodes_per_layer=64,
    n_hidden_layers=2,
    nf=max(1, int(NF_SECONDS / TS_NEW)),
    na_nb=max(1, int(NANB_SECONDS / TS_NEW)),
    batch_size=4000,
    lr=1e-4,
    epochs=50,
)

DIAG_EPOCHS = [0, 5, 10, 25, 50]

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
CH_NAMES    = ['X1', 'X2', 'Y']

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',           'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',        'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',    'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'


# =========================================================================
# Data loading
# =========================================================================

def get_data_dir(mode):
    base = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab')
    if mode == 'multisine':
        return os.path.join(base, 'multisine', 'baseline')
    else:
        return os.path.join(base, 'trajectories')


def _load_u(d):
    """Return plant input: 'u_total' for multisine, 'u' for trajectories."""
    if 'u_total' in d:
        return d['u_total']
    return d['u']


def load_traj(filename, data_dir):
    """Load .mat file, return (System_data, x_logical) both downsampled."""
    d = loadmat(os.path.join(data_dir, filename), squeeze_me=True)
    sys_data = deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    return sys_data, x_logical


# =========================================================================
# Normalisation (identical to gantry_interconnect_dynamic.py)
# =========================================================================

def compute_normalization(train_list):
    """Compute normalisation stats from training data."""
    u_all = np.concatenate([t.u for t in train_list])
    y_all = np.concatenate([t.y for t in train_list])

    fs = 1.0 / train_list[0].dt
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    x_logical_list = []
    for t in train_list:
        pos_logical = (P_inv_T @ t.y.T).T
        vel_logical = np.diff(pos_logical, axis=0) * fs
        vel_logical = np.vstack([vel_logical[:1], vel_logical])
        x_logical_list.append(np.hstack([pos_logical, vel_logical]))
    x_all = np.concatenate(x_logical_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np   = Dd.numpy()

    return dict(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0, Cd_norm=Cd_norm, Dd_np=Dd_np,
    )


# =========================================================================
# Build model (physics-only, no ANN)
# =========================================================================

def build_model(hp, norm, train_data):
    """Build physics-only interconnect with default learned encoder."""
    na = hp['na_nb']
    nb = hp['na_nb']

    ic = Interconnect(NX_PHYS, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=norm['std_x'], std_u=norm['std_u'],
        x_mean=norm['x_mean'], u_mean=norm['u_mean'], Ts=TS_NEW,
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=norm['Cd_norm'], D=norm['Dd_np'])

    ic.add_block(phy_block)
    ic.add_block(out_block)
    ic.connect_signals("x", phy_block)
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp")
    ic.connect_signals("x", out_block)
    ic.connect_block_signals(out_block, ["u"], ["y"])

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    fit_sys.norm.u0   = norm['u_mean'].flatten()
    fit_sys.norm.ustd = norm['std_u'].flatten()
    fit_sys.norm.y0   = norm['y0']
    fit_sys.norm.ystd = norm['ystd']

    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(DTYPE_PT)

    return fit_sys


# =========================================================================
# Encoder window extraction
# =========================================================================

def get_encoder_windows(fit_sys, sys_data, x_logical, hp, n_windows=2000):
    """Extract encoder input windows and corresponding x_logical ground truth.

    For each window at index t, the encoder sees u[t-nb:t] and y[t-na:t].
    The ground truth state is x_logical[t] (state at start of future window).
    """
    na = hp['na_nb']
    nb = hp['na_nb']
    nf = hp['nf']

    norm_data = fit_sys.norm.transform(sys_data)
    u_arr = np.ascontiguousarray(norm_data.u)
    y_arr = np.ascontiguousarray(norm_data.y)
    T = len(u_arr)
    start = max(na, nb)

    # Valid window indices: encoder needs na/nb history, rollout needs nf future
    t_indices = np.arange(start, T - nf)
    if len(t_indices) > n_windows:
        t_indices = np.random.choice(t_indices, n_windows, replace=False)

    # Vectorised window extraction via index arrays
    u_row_idx = t_indices[:, None] - nb + np.arange(nb)[None, :]  # (W, nb)
    y_row_idx = t_indices[:, None] - na + np.arange(na)[None, :]  # (W, na)

    all_uhist = u_arr[u_row_idx]                  # (W, nb, nu)
    all_yhist = y_arr[y_row_idx]                  # (W, na, ny)
    all_xlog  = x_logical[t_indices]              # (W, 6)

    return (
        torch.tensor(all_uhist, dtype=DTYPE_PT),
        torch.tensor(all_yhist, dtype=DTYPE_PT),
        all_xlog,
    )


# =========================================================================
# Evaluation
# =========================================================================

def evaluate_encoder_states(fit_sys, uhist, yhist, x_logical_gt, norm, label=''):
    """Compare encoder output to x_logical ground truth.

    Returns:
        x_enc: (B, 6) denormalised encoder states
        metrics: dict per state with mae, corr, scale
    """
    with torch.no_grad():
        x_enc_norm = fit_sys.encoder(uhist, yhist).numpy()

    x_enc = x_enc_norm * norm['std_x'].flatten() + norm['x_mean'].flatten()

    metrics = {}
    tag = f' [{label}]' if label else ''
    print(f'\n  Per-state comparison{tag}:')
    print(f'  {"State":<8} {"MAE":>12} {"corr":>10} {"scale":>10}')
    print(f'  {"-"*44}')

    for i in range(NX_PHYS):
        enc_i = x_enc[:, i]
        gt_i  = x_logical_gt[:, i]
        mae = float(np.abs(enc_i - gt_i).mean())
        if enc_i.std() > 1e-10 and gt_i.std() > 1e-10:
            corr = float(np.corrcoef(enc_i, gt_i)[0, 1])
        else:
            corr = 0.0
        scale = float(enc_i.std() / (gt_i.std() + 1e-10))
        metrics[STATE_NAMES[i]] = dict(mae=mae, corr=corr, scale=scale)
        print(f'  {STATE_NAMES[i]:<8} {mae:>12.6f} {corr:>10.4f} {scale:>10.4f}')

    return x_enc, metrics


def compute_correlation_matrix(x_enc, x_logical_gt):
    """6x6 correlation: encoder state i (row) vs physical state j (col)."""
    # Full 12x12 correlation, then slice the off-diagonal 6x6 block
    combined = np.hstack([x_enc, x_logical_gt])            # (B, 12)
    full_corr = np.corrcoef(combined, rowvar=False)         # (12, 12)
    return full_corr[:NX_PHYS, NX_PHYS:]                    # (6, 6)


def print_correlation_matrix(corr_mat):
    """Print the 6x6 correlation matrix."""
    print(f'\n  6x6 correlation matrix (encoder rows, physical cols):')
    header = '  {:>8s}'.format('') + ''.join(f'  {s:>8s}' for s in STATE_NAMES)
    print(header)
    print(f'  {"-"*(8 + 10*NX_PHYS)}')
    for i in range(NX_PHYS):
        row = ''.join(f'  {corr_mat[i, j]:>8.3f}' for j in range(NX_PHYS))
        print(f'  enc[{i}]  {row}')


def compute_sim_nrms(fit_sys, sys_data, norm):
    """Run full simulation and compute per-channel NRMS."""
    sim_result = fit_sys.apply_experiment(sys_data)
    cheat_n = sim_result.cheat_n
    y_hat = sim_result.y
    y_ref = sys_data.y
    nrms = np.sqrt(((y_hat[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0)) / norm['ystd']
    return nrms


# =========================================================================
# Plots
# =========================================================================

def plot_loss_convergence(epoch_id, loss_train, loss_val, save_path, mode=''):
    """Plot 1: train/val loss convergence."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(epoch_id, loss_val, 'C0', label='Val loss')
    ax.semilogy(epoch_id, loss_train, 'C1', linestyle='--', alpha=0.7, label='Train loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('sim-RMS')
    ax.set_title(f'Loss convergence, physics-only, default encoder ({mode})')
    ax.legend()
    ax.grid(True, which='both')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_mae_convergence(checkpoint_maes, save_path, mode=''):
    """Plot 2: per-state MAE vs epoch (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Per-state MAE convergence, encoder vs x_logical ({mode})', fontsize=13)

    epochs = sorted(checkpoint_maes.keys())
    for i in range(NX_PHYS):
        ax = axes[i // 3, i % 3]
        maes = [checkpoint_maes[ep][STATE_NAMES[i]]['mae'] for ep in epochs]
        ax.plot(epochs, maes, 'b.-', linewidth=1.5, markersize=8)
        ax.set_title(STATE_NAMES[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.grid(True)
        ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_scatter(x_enc, x_logical_gt, save_path, mode=''):
    """Plot 3: encoder vs analytical per state (2x3 grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Encoder state vs analytical ground truth, best epoch ({mode})', fontsize=13)

    for i in range(NX_PHYS):
        ax = axes[i // 3, i % 3]
        ax.scatter(x_logical_gt[:, i], x_enc[:, i], s=1, alpha=0.3)
        lims = [
            min(x_logical_gt[:, i].min(), x_enc[:, i].min()),
            max(x_logical_gt[:, i].max(), x_enc[:, i].max()),
        ]
        ax.plot(lims, lims, 'r--', linewidth=1)
        ax.set_xlabel(f'{STATE_NAMES[i]} (analytical)')
        ax.set_ylabel(f'{STATE_NAMES[i]} (encoder)')
        ax.set_title(STATE_NAMES[i])
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_correlation_matrix(corr_mat, save_path, mode=''):
    """Plot 4: 6x6 heatmap of encoder-physical state correlations."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    for i in range(NX_PHYS):
        for j in range(NX_PHYS):
            color = 'white' if abs(corr_mat[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr_mat[i, j]:.2f}',
                    ha='center', va='center', color=color, fontsize=10)

    ax.set_xticks(range(NX_PHYS))
    ax.set_xticklabels(STATE_NAMES)
    ax.set_yticks(range(NX_PHYS))
    ax.set_yticklabels([f'enc[{i}]' for i in range(NX_PHYS)])
    ax.set_xlabel('Physical state (x_logical)')
    ax.set_ylabel('Encoder output index')
    ax.set_title(f'Encoder vs physical state correlation ({mode})')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {save_path}')


# =========================================================================
# Experiment runner
# =========================================================================

def run_experiment(mode, hp):
    """Run full encoder state recovery experiment for one data mode."""
    print(f'\n{"="*80}')
    print(f'  ENCODER STATE RECOVERY: {mode.upper()}')
    print(f'{"="*80}')

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_dir = get_data_dir(mode)
    save_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', '..',
        'simulations', 'gantry_subnet', 'encoder_state_recovery',
    )
    os.makedirs(save_dir, exist_ok=True)

    # ── Load data ───────────────────────────────────────────────────────
    print(f'\nData dir: {data_dir}')
    train_pairs = [load_traj(f, data_dir) for f in TRAIN_FILES]
    train_list = [p[0] for p in train_pairs]
    train_data = deepSI.System_data_list(train_list)

    val_data, val_xlog = load_traj(VAL_FILE, data_dir)
    test_data, test_xlog = load_traj(TEST_FILE, data_dir)

    print(f'  {len(train_list)} train, 1 val ({val_data.u.shape[0]} samples), '
          f'1 test ({test_data.u.shape[0]} samples)')
    print(f'  x_logical: {val_xlog.shape}')

    # ── Normalisation ───────────────────────────────────────────────────
    norm = compute_normalization(train_list)

    # ── Config summary ──────────────────────────────────────────────────
    print(f'\nConfiguration:')
    print(f'  Mode:             {mode}')
    print(f'  Sampling rate:    {FS_NEW} Hz (D={D})')
    print(f'  nxd:              {NX_PHYS} (physics-only, no ANN)')
    print(f'  Encoder:          default (learned)')
    print(f'  nf:               {hp["nf"]} ({hp["nf"] * TS_NEW:.3f} s)')
    print(f'  na_nb:            {hp["na_nb"]} ({hp["na_nb"] * TS_NEW:.3f} s)')
    print(f'  Epochs:           {hp["epochs"]}')
    print(f'  Batch size:       {hp["batch_size"]}')
    print(f'  Learning rate:    {hp["lr"]}')
    print(f'  Dtype:            {"float64" if USE_F64 else "float32"}')

    # ── Build model ─────────────────────────────────────────────────────
    print('\nBuilding physics-only model with default encoder...')
    fit_sys = build_model(hp, norm, train_data)

    n_enc_params = sum(p.numel() for p in fit_sys.encoder.parameters())
    print(f'  Encoder parameters: {n_enc_params}')

    # ── Extract validation windows ──────────────────────────────────────
    print('\nExtracting encoder windows...')
    uhist_val, yhist_val, xlog_val = get_encoder_windows(
        fit_sys, val_data, val_xlog, hp)
    print(f'  Val: {uhist_val.shape[0]} windows')

    # ── Checkpoint training ─────────────────────────────────────────────
    diag_epochs = sorted(set([e for e in DIAG_EPOCHS if e <= hp['epochs']] + [hp['epochs']]))

    print(f'\nTraining with checkpoints at epochs: {diag_epochs}')
    checkpoint_maes = {}
    epochs_done = 0

    for target_epoch in diag_epochs:
        if target_epoch == 0:
            _, metrics = evaluate_encoder_states(
                fit_sys, uhist_val, yhist_val, xlog_val, norm,
                'epoch 0, random init')
            checkpoint_maes[0] = metrics
            continue

        epochs_to_run = target_epoch - epochs_done
        if epochs_to_run <= 0:
            continue

        print(f'\n  Training epochs {epochs_done + 1} to {target_epoch}...')
        fit_sys.fit(
            train_sys_data=train_data, val_sys_data=val_data,
            batch_size=hp['batch_size'], epochs=epochs_to_run,
            auto_fit_norm=False,
            loss_kwargs={'nf': hp['nf']},
            optimizer_kwargs={'lr': hp['lr']},
            validation_measure="sim-RMS",
        )
        epochs_done = target_epoch

        _, metrics = evaluate_encoder_states(
            fit_sys, uhist_val, yhist_val, xlog_val, norm,
            f'epoch {target_epoch}')
        checkpoint_maes[target_epoch] = metrics

    # ── Load best checkpoint, capture full loss history ──────────────────
    fit_sys.checkpoint_load_system(name='_last')
    epoch_id   = fit_sys.epoch_id.copy()
    loss_train = fit_sys.Loss_train.copy()
    loss_val   = fit_sys.Loss_val.copy()
    fit_sys.checkpoint_load_system(name='_best')
    fit_sys.eval()

    best_epoch = int(epoch_id[np.argmin(loss_val)])
    print(f'\n  Best epoch: {best_epoch} (val loss = {min(loss_val):.6f})')

    # ── Final evaluation: validation ────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  FINAL EVALUATION: Validation ({mode})')
    print(f'{"="*70}')

    x_enc_val, metrics_val = evaluate_encoder_states(
        fit_sys, uhist_val, yhist_val, xlog_val, norm, 'val, best')

    corr_mat_val = compute_correlation_matrix(x_enc_val, xlog_val)
    print_correlation_matrix(corr_mat_val)

    nrms_val = compute_sim_nrms(fit_sys, val_data, norm)
    print(f'\n  Sim-NRMS (val):')
    for ch, lbl in enumerate(CH_NAMES):
        print(f'    {lbl}: {nrms_val[ch]:.6f}')

    # ── Final evaluation: test ──────────────────────────────────────────
    print(f'\n{"="*70}')
    print(f'  FINAL EVALUATION: Test ({mode})')
    print(f'{"="*70}')

    uhist_test, yhist_test, xlog_test = get_encoder_windows(
        fit_sys, test_data, test_xlog, hp)
    x_enc_test, metrics_test = evaluate_encoder_states(
        fit_sys, uhist_test, yhist_test, xlog_test, norm, 'test, best')

    nrms_test = compute_sim_nrms(fit_sys, test_data, norm)
    print(f'\n  Sim-NRMS (test):')
    for ch, lbl in enumerate(CH_NAMES):
        print(f'    {lbl}: {nrms_test[ch]:.6f}')

    # ── Plots ───────────────────────────────────────────────────────────
    prefix = f'enc_recovery_{mode}_{run_id}'
    print(f'\nSaving plots...')

    plot_loss_convergence(
        epoch_id, loss_train, loss_val,
        os.path.join(save_dir, f'{prefix}_loss.png'), mode)

    plot_mae_convergence(
        checkpoint_maes,
        os.path.join(save_dir, f'{prefix}_mae_convergence.png'), mode)

    plot_scatter(
        x_enc_val, xlog_val,
        os.path.join(save_dir, f'{prefix}_scatter_val.png'), mode)

    plot_correlation_matrix(
        corr_mat_val,
        os.path.join(save_dir, f'{prefix}_corr_matrix.png'), mode)

    # ── Save results ────────────────────────────────────────────────────
    # Build checkpoint MAE array: (n_checkpoints, 6)
    ckpt_epochs_sorted = sorted(checkpoint_maes.keys())
    ckpt_mae_arr = np.array([
        [checkpoint_maes[ep][s]['mae'] for s in STATE_NAMES]
        for ep in ckpt_epochs_sorted
    ])

    npz_path = os.path.join(save_dir, f'{prefix}_results.npz')
    np.savez(npz_path,
        x_enc_val=x_enc_val,
        x_logical_val=xlog_val,
        x_enc_test=x_enc_test,
        x_logical_test=xlog_test,
        loss_train=loss_train,
        loss_val=loss_val,
        epoch_id=epoch_id,
        nrms_val=nrms_val,
        nrms_test=nrms_test,
        corr_matrix_val=corr_mat_val,
        checkpoint_epochs=np.array(ckpt_epochs_sorted),
        checkpoint_maes=ckpt_mae_arr,
        best_epoch=np.array(best_epoch),
        hp=json.dumps(hp),
        mode=mode,
    )
    print(f'  Saved: {npz_path}')

    return {
        'metrics_val': metrics_val,
        'metrics_test': metrics_test,
        'nrms_val': nrms_val,
        'nrms_test': nrms_test,
        'corr_matrix': corr_mat_val,
    }


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encoder state recovery test')
    parser.add_argument('--epochs', type=int, default=DEFAULT_HP['epochs'],
                        help='Training epochs per experiment')
    parser.add_argument('--lr', type=float, default=DEFAULT_HP['lr'],
                        help='Learning rate')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['multisine', 'trajectories', 'both'],
                        help='Data mode to run')
    args = parser.parse_args()

    hp = DEFAULT_HP.copy()
    hp['epochs'] = args.epochs
    hp['lr'] = args.lr

    print(f'Run ID: {run_id}')
    print(f'Hyperparameters:')
    for k, v in hp.items():
        print(f'  {k}: {v}')

    if args.mode in ('multisine', 'both'):
        run_experiment('multisine', hp)

    if args.mode in ('trajectories', 'both'):
        run_experiment('trajectories', hp)

    print(f'\n{"="*80}')
    print(f'  ALL EXPERIMENTS COMPLETE')
    print(f'{"="*80}')
