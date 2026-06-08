"""
diagnose_encoder.py
-------------------
Pre-training encoder diagnostic for the gantry SUBNET.

Builds a fresh SSE_Interconnect (identical to gantry_interconnect_dynamic.py),
runs a short training, then tests the encoder using multi-window rollouts
(matching the training loss setup). Compares encoder-initialised rollouts
against the analytical baseline (positions from measurements, velocities
from finite differences, ANN states = 0).

Diagnostics:
  1. Gradient flow: does loss.backward() produce non-zero encoder gradients?
  2. Short training: does loss decrease and do encoder weights change?
  3. Multi-window rollouts: encoder-init vs analytical-init (the baseline
     the augmentation structure must outperform)

Outputs (saved to simulations/gantry_subnet/encoder_verification/):
  All filenames include {run_id} (SLURM_JOB_ID or timestamp).
  - diag_encoder_results_{rid}.npz    : all data to reconstruct any plot
  - diag_training_curves_{rid}.png    : train/val loss vs epoch
  - diag_rollout_<traj>_{rid}.png     : per-trajectory time-domain rollouts
  - diag_per_channel_rms_{rid}.png    : per-channel (X1, X2, Y) bar chart
  - diag_error_over_time_{rid}.png    : instantaneous RMS error vs rollout step
  - diag_encoder_states_{rid}.png     : encoder x0 vs analytical x0, ANN state histogram
  - diag_encoder_model_{rid}          : trained model checkpoint

Usage:
  conda run -n GraduationProject python diagnose_encoder.py
  conda run -n GraduationProject python diagnose_encoder.py --epochs 20
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import (
    SSE_Interconnect, Interconnect, modified_encoder_net,
)
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.systems.gantry_ss import Cd, Dd, P

# =========================================================================
# Configuration (must match gantry_interconnect_dynamic.py)
# =========================================================================

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None
SEED = 42

FS_ORIG = 20000
FS_NEW  = 1000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

# Hyperparameters for diagnostic (same defaults as gantry_interconnect_dynamic.py)
DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=128,
    n_hidden_layers=3,
    nf=350,
    batch_size=4000,
    lr=7.6e-4,
    epochs=50,
)

# =========================================================================
# Data loading (identical to gantry_interconnect_dynamic.py)
# =========================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab', 'trajectories')

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=d['u'][::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data  = load_traj(VAL_FILE)
test_data = load_traj(TEST_FILE)

print(f'Loaded {len(train_list)} training trajectories, 1 val, 1 test')

# =========================================================================
# Normalisation (identical to gantry_interconnect_dynamic.py)
# =========================================================================

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])

fs = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)  # stage -> logical
x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t.y.T).T        # (N, 3) stage -> logical
    vel_logical = np.diff(pos_logical, axis=0) * fs  # (N-1, 3)
    vel_logical = np.vstack([vel_logical[:1], vel_logical])  # (N, 3)
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))  # (N, 6)
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()
PHY_IX  = np.arange(NX_PHYS)

save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet', 'encoder_verification')
os.makedirs(save_dir, exist_ok=True)


# =========================================================================
# Build model (identical to build_and_train, but WITHOUT calling fit)
# =========================================================================

def build_model(hp):
    """Build a fresh SSE_Interconnect with manual normalisation. Does NOT train."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    na = 2 * nxd + 1
    nb = 2 * nxd + 1

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
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
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    return fit_sys


# =========================================================================
# Analytical baseline x0
# =========================================================================

def analytical_x0_at(y_seq, t_idx, dt):
    """
    Compute analytical physical state at time index t_idx.

    Converts stage coordinates (X1, X2, Y) to logical coordinates
    (q1, q2, q3) via P^{-T}, then computes velocities from finite
    differences in logical space.

    Returns: (6,) unnormalized physical state in logical coordinates.
    """
    pos_stage = y_seq[t_idx]
    pos = P_inv_T @ pos_stage
    if 0 < t_idx < len(y_seq) - 1:
        vel_stage = (y_seq[t_idx + 1] - y_seq[t_idx - 1]) / (2 * dt)
    elif t_idx == 0:
        vel_stage = (y_seq[t_idx + 1] - y_seq[t_idx]) / dt
    else:
        vel_stage = (y_seq[t_idx] - y_seq[t_idx - 1]) / dt
    vel = P_inv_T @ vel_stage
    return np.concatenate([pos, vel])


def normalize_x(x_phys):
    """Normalize physical state vector (6,) to model's internal representation."""
    return ((x_phys.reshape(NX_PHYS, 1) - x_mean) / std_x).flatten()


# =========================================================================
# Diagnostic helpers
# =========================================================================

def get_encoder_x0(fit_sys, data_norm, t_start):
    """Run encoder on a single window ending at t_start. Returns (nxd,) numpy."""
    na, nb = fit_sys.na, fit_sys.nb
    uhist = np.ascontiguousarray(data_norm.u[t_start - nb:t_start]).reshape(1, nb, nu)
    yhist = np.ascontiguousarray(data_norm.y[t_start - na:t_start]).reshape(1, na, ny)
    with torch.no_grad():
        x0 = fit_sys.encoder(
            torch.tensor(uhist, dtype=DTYPE_PT),
            torch.tensor(yhist, dtype=DTYPE_PT),
        )
    return x0.squeeze(0).numpy()


def rollout_from_x0(fit_sys, data_norm, x0_norm, t_start, n_steps,
                    return_states=False):
    """Simulate n_steps from normalized x0.

    Returns:
        y_hat: (n_steps, 3) physical output
        x_traj: (n_steps, nxd) normalized states (only if return_states=True)
    """
    x = torch.tensor(x0_norm.reshape(1, -1), dtype=DTYPE_PT)
    u_norm = torch.tensor(
        np.ascontiguousarray(data_norm.u[t_start:t_start + n_steps]),
        dtype=DTYPE_PT,
    )
    y_list = []
    x_list = []
    with torch.no_grad():
        for t in range(min(n_steps, len(u_norm))):
            y_t, x = fit_sys.hfn(x, u_norm[t:t + 1])
            y_list.append(y_t.squeeze(0).numpy())
            if return_states:
                x_list.append(x.squeeze(0).numpy())
    y_hat_norm = np.array(y_list)
    y_hat = y_hat_norm * ystd + fit_sys.norm.y0
    if return_states:
        return y_hat, np.array(x_list)
    return y_hat


# =========================================================================
# DIAGNOSTIC 1: Gradient flow
# =========================================================================

def check_gradient_flow(fit_sys, hp):
    print(f"\n{'='*70}")
    print("DIAGNOSTIC 1: Gradient flow check")
    print(f"{'='*70}")

    nf = hp['nf']
    val_norm = fit_sys.norm.transform(val_data)

    # Build one training window (same as loss() receives)
    uhist, yhist, ufuture, yfuture = val_norm.to_hist_future_data(
        na=fit_sys.na, nb=fit_sys.nb, nf=nf,
    )
    # Take a small batch
    idx = np.random.choice(len(uhist), size=min(4, len(uhist)), replace=False)
    uh = torch.tensor(uhist[idx], dtype=DTYPE_PT)
    yh = torch.tensor(yhist[idx], dtype=DTYPE_PT)
    uf = torch.tensor(ufuture[idx], dtype=DTYPE_PT)
    yf = torch.tensor(yfuture[idx], dtype=DTYPE_PT)

    # Zero all gradients
    fit_sys.optimizer.zero_grad()

    # Forward + backward
    loss = fit_sys.loss(uh, yh, uf, yf)
    loss.backward()

    # Check encoder gradients
    enc_params = list(fit_sys.encoder.parameters())
    n_total = sum(p.numel() for p in enc_params)
    n_with_grad = sum(p.numel() for p in enc_params if p.grad is not None and p.grad.abs().sum() > 0)
    grad_norms = [p.grad.norm().item() for p in enc_params if p.grad is not None]

    print(f"  Loss value:               {loss.item():.6f}")
    print(f"  Encoder parameters:       {n_total}")
    print(f"  Parameters with gradient: {n_with_grad}")
    if grad_norms:
        print(f"  Gradient norm (min/max):  {min(grad_norms):.2e} / {max(grad_norms):.2e}")

    if n_with_grad == 0:
        print("\n  ** FAIL: No gradients reach the encoder. Training cannot update it. **")
        return False
    elif n_with_grad < n_total:
        print(f"\n  ** WARNING: Only {n_with_grad}/{n_total} parameters have gradient. **")
        return True
    else:
        print("\n  PASS: Gradients flow through to all encoder parameters.")
        return True


# =========================================================================
# DIAGNOSTIC 2: Short training
# =========================================================================

def short_training(fit_sys, hp):
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 2: Short training ({hp['epochs']} epochs)")
    print(f"{'='*70}")

    # Snapshot encoder weights before training
    w_before = {n: p.clone().detach() for n, p in fit_sys.encoder.named_parameters()}

    fit_sys.fit(
        train_sys_data=train_data, val_sys_data=val_data,
        batch_size=hp['batch_size'], epochs=hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': hp['nf']},
        optimizer_kwargs={'lr': hp['lr']},
        validation_measure="sim-RMS",
    )

    # fit() ends with _best loaded, which truncates history arrays.
    # Load _last to get full training history, then reload _best.
    fit_sys.checkpoint_load_system(name='_last')
    loss_train_full = fit_sys.Loss_train.copy()
    loss_val_full   = fit_sys.Loss_val.copy()
    epoch_id_full   = fit_sys.epoch_id.copy()
    batch_id_full   = fit_sys.batch_id.copy()
    time_full       = fit_sys.time.copy()
    fit_sys.checkpoint_load_system(name='_best')
    fit_sys.eval()

    best_epoch_idx = int(np.argmin(loss_val_full))

    # Compare encoder weights after training
    w_after = {n: p.clone().detach() for n, p in fit_sys.encoder.named_parameters()}

    print(f"\n  Encoder weight changes after {hp['epochs']} epochs:")
    print(f"  {'Layer':<30s}  {'L2 change':>12s}  {'Relative':>12s}")
    total_change = 0.0
    for name in w_before:
        diff = (w_after[name] - w_before[name]).norm().item()
        w_norm = w_before[name].norm().item()
        rel = diff / max(w_norm, 1e-8)
        total_change += diff
        print(f"  {name:<30s}  {diff:12.6f}  {rel:12.6f}")

    train_ok = total_change > 1e-10
    if not train_ok:
        print("\n  ** FAIL: Encoder weights did not change. Training is not updating it. **")
    else:
        print(f"\n  PASS: Total weight change = {total_change:.6f}")

    return {
        'train_ok': train_ok,
        'loss_train': loss_train_full,
        'loss_val': loss_val_full,
        'epoch_id': epoch_id_full,
        'batch_id': batch_id_full,
        'time': time_full,
        'best_epoch_idx': best_epoch_idx,
        'bestfit': float(loss_val_full[best_epoch_idx]),
    }


# =========================================================================
# DIAGNOSTIC 3: Multi-window rollouts (encoder vs analytical baseline)
# =========================================================================

def multi_window_rollouts(fit_sys, hp):
    """Collect per-window rollout data for encoder vs analytical baseline.

    Model must already be at best checkpoint (done by short_training).
    Returns a dict with all rollout data for saving and plotting.
    """
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    nf = hp['nf']
    na, nb = fit_sys.na, fit_sys.nb
    cheat_n = max(na, nb)

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 3: Multi-window rollouts (nf={nf} steps = {nf*TS_NEW:.3f} s)")
    print(f"{'='*70}")
    print(f"  Comparing encoder-init vs analytical baseline (pos + finite-diff vel, ANN=0)")

    all_data = train_list + [val_data, test_data]
    all_labels = TRAIN_FILES + [VAL_FILE, TEST_FILE]
    all_norms = [fit_sys.norm.transform(d) for d in all_data]

    rms_encoder_all = []
    rms_analytical_all = []
    trajectories = []

    print(f"\n  {'Trajectory':<40s}  {'Windows':>7s}  "
          f"{'Enc RMS':>10s}  {'Ana RMS':>10s}  {'Winner':>8s}")

    for traj_idx, (data_raw, data_norm, label) in enumerate(
            zip(all_data, all_norms, all_labels)):

        T = len(data_raw.u)
        if T < cheat_n + nf:
            print(f"  {label:<40s}  SKIPPED (too short)")
            trajectories.append({
                'label': label, 'skipped': True,
                'starts': np.array([]), 'windows': [],
            })
            continue

        usable_start = cheat_n
        usable_end = T - nf
        n_windows = min(10, (usable_end - usable_start) // nf)
        if n_windows < 1:
            n_windows = 1
        starts = np.linspace(usable_start, usable_end, n_windows, dtype=int)

        traj_rms_enc = []
        traj_rms_ana = []
        windows = []

        for t_s in starts:
            x0_enc = get_encoder_x0(fit_sys, data_norm, t_s)
            y_enc, x_traj_enc = rollout_from_x0(
                fit_sys, data_norm, x0_enc, t_s, nf, return_states=True)

            x0_ana_phys = analytical_x0_at(data_raw.y, t_s, TS_NEW)
            x0_ana_norm = normalize_x(x0_ana_phys)
            x0_ana_full = np.zeros(nxd, dtype=DTYPE_NP)
            x0_ana_full[:NX_PHYS] = x0_ana_norm
            y_ana, x_traj_ana = rollout_from_x0(
                fit_sys, data_norm, x0_ana_full, t_s, nf, return_states=True)

            n_cmp = min(nf, len(data_raw.y) - t_s, len(y_enc))
            y_ref = data_raw.y[t_s:t_s + n_cmp]

            rms_e = np.sqrt(np.mean((y_enc[:n_cmp] - y_ref) ** 2))
            rms_a = np.sqrt(np.mean((y_ana[:n_cmp] - y_ref) ** 2))
            rms_enc_ch = np.sqrt(np.mean((y_enc[:n_cmp] - y_ref) ** 2, axis=0))
            rms_ana_ch = np.sqrt(np.mean((y_ana[:n_cmp] - y_ref) ** 2, axis=0))

            traj_rms_enc.append(rms_e)
            traj_rms_ana.append(rms_a)

            windows.append({
                'y_ref': y_ref.copy(),
                'y_enc': y_enc[:n_cmp].copy(),
                'y_ana': y_ana[:n_cmp].copy(),
                'x_traj_enc': x_traj_enc[:n_cmp].copy(),
                'x_traj_ana': x_traj_ana[:n_cmp].copy(),
                'x0_enc': x0_enc.copy(),
                'x0_ana': x0_ana_full.copy(),
                'rms_enc_ch': rms_enc_ch,
                'rms_ana_ch': rms_ana_ch,
                'rms_enc': rms_e,
                'rms_ana': rms_a,
                't_start': int(t_s),
                'n_cmp': n_cmp,
            })

        mean_e = np.mean(traj_rms_enc)
        mean_a = np.mean(traj_rms_ana)
        rms_encoder_all.extend(traj_rms_enc)
        rms_analytical_all.extend(traj_rms_ana)

        trajectories.append({
            'label': label,
            'skipped': False,
            'starts': starts,
            'windows': windows,
            'mean_rms_enc': mean_e,
            'mean_rms_ana': mean_a,
        })

        winner = "Encoder" if mean_e < mean_a else "Baseline"
        print(f"  {label:<40s}  {len(starts):>7d}  "
              f"{mean_e:10.6f}  {mean_a:10.6f}  {winner:>8s}")

    overall_enc = np.mean(rms_encoder_all)
    overall_ana = np.mean(rms_analytical_all)
    print(f"\n  {'OVERALL':<40s}  {len(rms_encoder_all):>7d}  "
          f"{overall_enc:10.6f}  {overall_ana:10.6f}")

    if overall_enc < overall_ana:
        pct = (1 - overall_enc / overall_ana) * 100
        print(f"\n  Encoder OUTPERFORMS analytical baseline by {pct:.1f}%.")
        print(f"  Augmentation structure is adding value.")
    else:
        pct = (overall_enc / overall_ana - 1) * 100
        print(f"\n  Encoder is WORSE than analytical baseline by {pct:.1f}%.")
        print(f"  Augmentation structure is not helping after {hp['epochs']} epochs.")

    return {
        'overall_enc': overall_enc,
        'overall_ana': overall_ana,
        'trajectories': trajectories,
        'nf': nf,
        'cheat_n': cheat_n,
    }


# =========================================================================
# Plot functions
# =========================================================================

CH_LABELS = ['X1 [m]', 'X2 [m]', 'Y [m]']


def plot_training_curves(train_result, save_dir):
    """Plot 1: training and validation loss vs epoch."""
    epoch_id = train_result['epoch_id']
    loss_val = train_result['loss_val']
    loss_train = train_result['loss_train']
    best_idx = train_result['best_epoch_idx']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.semilogy(epoch_id, loss_val, 'C0', label='Val sim-RMS')
    ax.semilogy(epoch_id, loss_train, 'C1--', alpha=0.7, label='Train loss')
    ax.axvline(epoch_id[best_idx], color='red', linestyle=':', alpha=0.7,
               label=f'Best epoch {epoch_id[best_idx]:.0f} (val={loss_val[best_idx]:.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_training_curves_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_per_trajectory_rollouts(rollout_result, save_dir):
    """Plot 2: time-domain rollouts per trajectory (one figure each)."""
    for traj in rollout_result['trajectories']:
        if traj['skipped'] or not traj['windows']:
            continue

        label = traj['label']
        windows = traj['windows']
        n_win = len(windows)
        label_stem = os.path.splitext(label)[0]

        fig, axes = plt.subplots(3, n_win, figsize=(3.5 * n_win, 7),
                                 squeeze=False, sharex='col')

        for col, win in enumerate(windows):
            n_cmp = win['n_cmp']
            t_plot = np.arange(n_cmp) * TS_NEW

            for ch in range(3):
                ax = axes[ch, col]
                ax.plot(t_plot, win['y_ref'][:, ch], 'k', lw=1, label='Reference')
                ax.plot(t_plot, win['y_enc'][:, ch], 'C0', lw=0.8, label='Encoder')
                ax.plot(t_plot, win['y_ana'][:, ch], 'C1--', lw=0.8, label='Analytical')
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel(CH_LABELS[ch], fontsize=8)
                if ch == 0:
                    ax.set_title(f't0={win["t_start"]*TS_NEW:.2f}s', fontsize=8)
                    if col == 0:
                        ax.legend(fontsize=6)
                if ch == 2:
                    ax.set_xlabel('Time [s]', fontsize=7)

        fig.suptitle(f'{label_stem}: encoder vs analytical baseline', fontsize=10)
        fig.tight_layout()
        path = os.path.join(save_dir, f'diag_rollout_{label_stem}_{run_id}.png')
        fig.savefig(path, dpi=150)
        print(f"  Saved: {os.path.basename(path)}")


def plot_per_channel_bar_chart(rollout_result, hp, save_dir):
    """Plot 3: per-channel RMS bar chart (X1, X2, Y separately)."""
    trajs = rollout_result['trajectories']
    active = [t for t in trajs if not t['skipped']]
    labels = [os.path.splitext(t['label'])[0] for t in active]

    # Compute per-channel means for each trajectory
    means_enc = np.zeros((len(active), 3))
    means_ana = np.zeros((len(active), 3))
    for i, t in enumerate(active):
        enc_ch = np.array([w['rms_enc_ch'] for w in t['windows']])
        ana_ch = np.array([w['rms_ana_ch'] for w in t['windows']])
        means_enc[i] = enc_ch.mean(axis=0)
        means_ana[i] = ana_ch.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    x_pos = np.arange(len(active))
    bar_w = 0.35

    for ch in range(3):
        ax = axes[ch]
        ax.bar(x_pos - bar_w/2, means_enc[:, ch], bar_w,
               label='Encoder', color='C0')
        ax.bar(x_pos + bar_w/2, means_ana[:, ch], bar_w,
               label='Analytical', color='C1')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_title(CH_LABELS[ch])
        ax.set_ylabel('Mean RMS [m]')
        ax.grid(True, alpha=0.3, axis='y')
        if ch == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Per-channel RMS: encoder vs analytical ({hp["epochs"]} epochs)',
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_per_channel_rms_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_error_over_time(rollout_result, save_dir):
    """Plot 4: instantaneous RMS error vs rollout timestep, averaged over all windows."""
    nf = rollout_result['nf']

    # Collect per-timestep squared errors from all windows
    err_enc_list = []
    err_ana_list = []
    for traj in rollout_result['trajectories']:
        if traj['skipped']:
            continue
        for win in traj['windows']:
            n = win['n_cmp']
            # Pad to nf if shorter (edge windows)
            e_enc = np.full((nf, 3), np.nan)
            e_ana = np.full((nf, 3), np.nan)
            e_enc[:n] = (win['y_enc'] - win['y_ref']) ** 2
            e_ana[:n] = (win['y_ana'] - win['y_ref']) ** 2
            err_enc_list.append(e_enc)
            err_ana_list.append(e_ana)

    err_enc = np.array(err_enc_list)  # (n_windows, nf, 3)
    err_ana = np.array(err_ana_list)

    # RMS across windows and channels at each timestep
    rms_enc_t = np.sqrt(np.nanmean(err_enc, axis=(0, 2)))  # (nf,)
    rms_ana_t = np.sqrt(np.nanmean(err_ana, axis=(0, 2)))

    # Also per-channel
    rms_enc_t_ch = np.sqrt(np.nanmean(err_enc, axis=0))  # (nf, 3)
    rms_ana_t_ch = np.sqrt(np.nanmean(err_ana, axis=0))

    t_axis = np.arange(nf) * TS_NEW

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: overall
    ax = axes[0]
    ax.plot(t_axis, rms_enc_t, 'C0', lw=1, label='Encoder')
    ax.plot(t_axis, rms_ana_t, 'C1', lw=1, label='Analytical')
    ax.set_ylabel('RMS error [m]')
    ax.set_title('Instantaneous RMS error vs rollout time (all windows)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom: per-channel
    ax = axes[1]
    for ch in range(3):
        ax.plot(t_axis, rms_enc_t_ch[:, ch], f'C{ch}', lw=0.8,
                label=f'{CH_LABELS[ch]} enc')
        ax.plot(t_axis, rms_ana_t_ch[:, ch], f'C{ch}--', lw=0.8,
                label=f'{CH_LABELS[ch]} ana')
    ax.set_xlabel('Rollout time [s]')
    ax.set_ylabel('RMS error [m]')
    ax.set_title('Per-channel breakdown')
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_error_over_time_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_encoder_state_inspection(rollout_result, hp, save_dir):
    """Plot 5: encoder x0 vs analytical x0 (physical states), ANN state histogram."""
    NX_ANN = hp['NX_ANN']

    # Collect all x0 pairs
    x0_enc_all = []
    x0_ana_all = []
    for traj in rollout_result['trajectories']:
        if traj['skipped']:
            continue
        for win in traj['windows']:
            x0_enc_all.append(win['x0_enc'])
            x0_ana_all.append(win['x0_ana'])

    x0_enc_all = np.array(x0_enc_all)  # (N, nxd)
    x0_ana_all = np.array(x0_ana_all)

    state_labels = ['X1', 'X2', 'Y', 'dX1', 'dX2', 'dY']

    # Layout: 2 rows of 3 for physical states, then 1 row for ANN histogram
    fig = plt.figure(figsize=(14, 9))

    # Physical states: encoder vs analytical scatter
    for i in range(NX_PHYS):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.scatter(x0_ana_all[:, i], x0_enc_all[:, i], s=12, alpha=0.6, c='C0')
        lims = [
            min(x0_ana_all[:, i].min(), x0_enc_all[:, i].min()),
            max(x0_ana_all[:, i].max(), x0_enc_all[:, i].max()),
        ]
        span = lims[1] - lims[0]
        margin = span * 0.05 if span > 0 else 0.1
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', lw=0.5, alpha=0.5)
        ax.set_xlabel('Analytical', fontsize=7)
        ax.set_ylabel('Encoder', fontsize=7)
        ax.set_title(f'{state_labels[i]} (normalized)', fontsize=8)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    # ANN states histogram
    for j in range(NX_ANN):
        ax = fig.add_subplot(3, 3, NX_PHYS + 1 + j)
        ann_vals = x0_enc_all[:, NX_PHYS + j]
        ax.hist(ann_vals, bins=30, color='C2', alpha=0.7, edgecolor='C2')
        ax.axvline(0, color='k', linestyle='--', lw=0.5, alpha=0.5)
        ax.set_xlabel(f'ANN state {j+1} value', fontsize=7)
        ax.set_ylabel('Count', fontsize=7)
        ax.set_title(f'ANN state {j+1} (encoder x0)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)

    fig.suptitle('Encoder state inspection: physical states (top) + ANN states (bottom)',
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_encoder_states_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


# =========================================================================
# Save all data to .npz
# =========================================================================

def save_diagnostic_npz(train_result, rollout_result, hp, save_dir):
    """Save all diagnostic data to a single .npz for later reconstruction."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    save_dict = {
        'hp_json': json.dumps(hp),
        'seed': SEED,
        'NX_PHYS': NX_PHYS,
        'NX_ANN': NX_ANN,
        'nxd': nxd,
        'TS_NEW': TS_NEW,
        'nf': rollout_result['nf'],
        'cheat_n': rollout_result['cheat_n'],
        'overall_rms_enc': rollout_result['overall_enc'],
        'overall_rms_ana': rollout_result['overall_ana'],
        # Training history
        'loss_train': train_result['loss_train'],
        'loss_val': train_result['loss_val'],
        'epoch_id': train_result['epoch_id'],
        'batch_id': train_result['batch_id'],
        'time': train_result['time'],
        'best_epoch_idx': train_result['best_epoch_idx'],
        'bestfit': train_result['bestfit'],
        # Normalization (for denormalization of saved states)
        'x_mean': x_mean,
        'std_x': std_x,
        'ystd': ystd,
        'y0': y0,
        # Trajectory metadata
        'n_trajectories': len(rollout_result['trajectories']),
    }

    for i, traj in enumerate(rollout_result['trajectories']):
        save_dict[f'traj{i}_label'] = traj['label']
        save_dict[f'traj{i}_skipped'] = traj['skipped']

        if traj['skipped']:
            continue

        save_dict[f'traj{i}_starts'] = traj['starts']
        save_dict[f'traj{i}_mean_rms_enc'] = traj['mean_rms_enc']
        save_dict[f'traj{i}_mean_rms_ana'] = traj['mean_rms_ana']
        save_dict[f'traj{i}_n_windows'] = len(traj['windows'])

        for j, win in enumerate(traj['windows']):
            p = f'traj{i}_win{j}_'
            save_dict[p + 'y_ref'] = win['y_ref']
            save_dict[p + 'y_enc'] = win['y_enc']
            save_dict[p + 'y_ana'] = win['y_ana']
            save_dict[p + 'x_traj_enc'] = win['x_traj_enc']
            save_dict[p + 'x_traj_ana'] = win['x_traj_ana']
            save_dict[p + 'x0_enc'] = win['x0_enc']
            save_dict[p + 'x0_ana'] = win['x0_ana']
            save_dict[p + 'rms_enc_ch'] = win['rms_enc_ch']
            save_dict[p + 'rms_ana_ch'] = win['rms_ana_ch']
            save_dict[p + 't_start'] = win['t_start']
            save_dict[p + 'n_cmp'] = win['n_cmp']

    path = os.path.join(save_dir, f'diag_encoder_results_{run_id}.npz')
    np.savez(path, **save_dict)
    print(f"  Saved: {os.path.basename(path)}")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encoder diagnostic for gantry SUBNET')
    parser.add_argument('--epochs', type=int, default=DEFAULT_HP['epochs'],
                        help=f'Short training epochs (default: {DEFAULT_HP["epochs"]})')
    args = parser.parse_args()

    hp = DEFAULT_HP.copy()
    hp['epochs'] = args.epochs

    print(f"\nBuilding fresh model with hp: {hp}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    fit_sys = build_model(hp)

    # Trigger encoder initialisation (normally happens lazily in fit())
    fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
    for net in (fit_sys.encoder, fit_sys.fn, fit_sys.hn):
        net.to(DTYPE_PT)
    print(f"Encoder initialised: {sum(p.numel() for p in fit_sys.encoder.parameters())} parameters")

    # Diagnostic 1: gradient flow
    grad_ok = check_gradient_flow(fit_sys, hp)
    if not grad_ok:
        print("\nAborting: no point training if gradients don't reach the encoder.")
        sys.exit(1)

    # Diagnostic 2: short training (captures history, loads best checkpoint)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_result = short_training(fit_sys, hp)
    train_ok = train_result['train_ok']

    # Plot 1: training curves
    plot_training_curves(train_result, save_dir)

    # Diagnostic 3: multi-window rollouts (model already at _best)
    rollout_result = multi_window_rollouts(fit_sys, hp)

    # Plots 2-5
    plot_per_trajectory_rollouts(rollout_result, save_dir)
    plot_per_channel_bar_chart(rollout_result, hp, save_dir)
    plot_error_over_time(rollout_result, save_dir)
    plot_encoder_state_inspection(rollout_result, hp, save_dir)

    # Save all data
    save_diagnostic_npz(train_result, rollout_result, hp, save_dir)

    # Save trained model checkpoint
    model_path = os.path.join(save_dir, f'diag_encoder_model_{run_id}')
    fit_sys.save_system(model_path)
    print(f"  Saved model: {model_path}")

    plt.close('all')

    # Final verdict
    rms_enc = rollout_result['overall_enc']
    rms_ana = rollout_result['overall_ana']
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if not grad_ok:
        print("  FAIL: Gradients do not reach encoder.")
    elif not train_ok:
        print("  FAIL: Encoder weights unchanged after training.")
    elif rms_enc < rms_ana:
        print(f"  PASS: Encoder ({rms_enc:.6f}) beats analytical baseline ({rms_ana:.6f}).")
    else:
        print(f"  NOT YET: Encoder ({rms_enc:.6f}) does not beat baseline ({rms_ana:.6f}).")
        print(f"  This may improve with more epochs or different hyperparameters.")
