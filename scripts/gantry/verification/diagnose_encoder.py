"""
diagnose_encoder.py
-------------------
Pre-training encoder diagnostic for the gantry SUBNET.

Builds one or two SSE_Interconnect models with different encoder types,
runs short training, then tests using multi-window rollouts.

Encoder modes:
  - default: learned encoder (modified_encoder_net)
  - hybrid:  analytical physical states + learned augmented states
  - both:    runs both in parallel (one process each) for side-by-side comparison

Diagnostics per encoder:
  1. Gradient flow: does loss.backward() produce non-zero encoder gradients?
  2. Short training: does loss decrease and do encoder weights change?
  3. Multi-window rollouts: three-way comparison
     - Encoder: trained augmented model, encoder x0
     - Analytical (trained ANN): trained augmented model, analytical x0, ANN=0
     - Physics only: physics-only model (no ANN), analytical x0

When running both encoders, additional comparison plots are generated.

Outputs (saved to simulations/gantry_subnet/encoder_verification/):
  Per-encoder: diag_{diagnostic}_{encoder_mode}_{rid}.png
  Comparison:  diag_cmp_{diagnostic}_{rid}.png

Usage:
  python diagnose_encoder.py                          # both encoders, default epochs
  python diagnose_encoder.py --encoder hybrid         # hybrid only
  python diagnose_encoder.py --encoder both --epochs 20 --cpus-per-worker 4
"""

import os
import sys
import json
import argparse
import multiprocessing
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
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn, HybridGantryEncoder
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

# Data source: 'trajectories' or 'multisine'
MODE = 'trajectories'

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None
SEED = 42

FS_ORIG = 20000
FS_NEW  = 1000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

N_CPUS_PER_WORKER = 4  # torch intra-op threads per encoder process

run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

# Hyperparameters for diagnostic (matches gantry_interconnect_dynamic.py)
DEFAULT_HP = dict(
    NX_ANN=2,
    n_nodes_per_layer=64,
    n_hidden_layers=2,
    nf=350,
    batch_size=4000,
    lr=5e-4,
    epochs=50,
)

# =========================================================================
# Data loading (identical to gantry_interconnect_dynamic.py)
# =========================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab', DATA_SUBDIR)

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'

def _load_u(d):
    """Return plant input: 'u_total' for multisine data, 'u' for trajectory data."""
    if 'u_total' in d:
        return d['u_total']
    return d['u']

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
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

def build_model(hp, encoder_mode='default'):
    """Build SSE_Interconnect. encoder_mode: 'default' or 'hybrid'."""
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

    if encoder_mode == 'hybrid':
        fit_sys.encoder = HybridGantryEncoder(
            nb=nb, nu=nu, na=na, ny=ny, nx=nxd,
            P_inv_T=P_inv_T, y0=y0, ystd=ystd,
            x_mean=x_mean.flatten(), std_x=std_x.flatten(),
            fs=fs, NX_PHYS=NX_PHYS,
            n_nodes_per_layer=hp['n_nodes_per_layer'],
            n_hidden_layers=hp['n_hidden_layers'],
        ).to(DTYPE_PT)

    return fit_sys


def build_physics_only():
    """Build a physics-only interconnect (nxd=6, no ANN) for baseline comparison."""
    ic = Interconnect(NX_PHYS, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)

    ic.add_block(phy_block)
    ic.add_block(out_block)
    ic.connect_signals("x", phy_block)
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp")
    ic.connect_signals("x", out_block)
    ic.connect_block_signals(out_block, ["u"], ["y"])

    return ic


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


def rollout_from_x0(interconnect, data_norm, x0_norm, t_start, n_steps,
                    return_states=False):
    """Simulate n_steps from normalized x0 through the given interconnect.

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
            y_t, x = interconnect(x, u_norm[t:t + 1])
            y_list.append(y_t.squeeze(0).numpy())
            if return_states:
                x_list.append(x.squeeze(0).numpy())
    y_hat_norm = np.array(y_list)
    y_hat = y_hat_norm * ystd + y0
    if return_states:
        return y_hat, np.array(x_list)
    return y_hat


# =========================================================================
# DIAGNOSTIC 1: Gradient flow
# =========================================================================

def check_gradient_flow(fit_sys, hp, label=''):
    tag = f' [{label}]' if label else ''
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 1: Gradient flow check{tag}")
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

def short_training(fit_sys, hp, label=''):
    tag = f' [{label}]' if label else ''
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 2: Short training ({hp['epochs']} epochs){tag}")
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

def multi_window_rollouts(fit_sys, ic_phy, hp, label=''):
    """Collect per-window rollout data: encoder vs analytical (trained ANN) vs physics only.

    Model must already be at best checkpoint (done by short_training).
    Returns a dict with all rollout data for saving and plotting.
    """
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    nf = hp['nf']
    na, nb = fit_sys.na, fit_sys.nb
    cheat_n = max(na, nb)

    tag = f' [{label}]' if label else ''
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 3: Multi-window rollouts (nf={nf} steps = {nf*TS_NEW:.3f} s){tag}")
    print(f"{'='*70}")
    print(f"  Three-way comparison:")
    print(f"    Enc = trained augmented model, encoder x0")
    print(f"    Ana = trained augmented model, analytical x0 (ANN=0)")
    print(f"    Phy = physics-only model (no ANN), analytical x0")

    all_data = train_list + [val_data, test_data]
    all_labels = TRAIN_FILES + [VAL_FILE, TEST_FILE]
    all_norms = [fit_sys.norm.transform(d) for d in all_data]

    rms_encoder_all = []
    rms_analytical_all = []
    rms_physics_all = []
    trajectories = []

    print(f"\n  {'Trajectory':<40s}  {'Win':>4s}  "
          f"{'Enc RMS':>10s}  {'Ana RMS':>10s}  {'Phy RMS':>10s}  {'Winner':>8s}")

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
        traj_rms_phy = []
        windows = []

        for t_s in starts:
            # Encoder rollout (trained augmented model)
            x0_enc = get_encoder_x0(fit_sys, data_norm, t_s)
            y_enc, x_traj_enc = rollout_from_x0(
                fit_sys.hfn, data_norm, x0_enc, t_s, nf, return_states=True)

            # Analytical x0 in logical coordinates
            x0_ana_phys = analytical_x0_at(data_raw.y, t_s, TS_NEW)
            x0_ana_norm = normalize_x(x0_ana_phys)

            # Analytical rollout through trained augmented model (ANN=0)
            x0_ana_full = np.zeros(nxd, dtype=DTYPE_NP)
            x0_ana_full[:NX_PHYS] = x0_ana_norm
            y_ana, x_traj_ana = rollout_from_x0(
                fit_sys.hfn, data_norm, x0_ana_full, t_s, nf, return_states=True)

            # Physics-only rollout (no ANN, nxd=6)
            x0_phy = x0_ana_norm.copy()
            y_phy, x_traj_phy = rollout_from_x0(
                ic_phy, data_norm, x0_phy, t_s, nf, return_states=True)

            n_cmp = min(nf, len(data_raw.y) - t_s, len(y_enc))
            y_ref = data_raw.y[t_s:t_s + n_cmp]

            rms_e = np.sqrt(np.mean((y_enc[:n_cmp] - y_ref) ** 2))
            rms_a = np.sqrt(np.mean((y_ana[:n_cmp] - y_ref) ** 2))
            rms_p = np.sqrt(np.mean((y_phy[:n_cmp] - y_ref) ** 2))
            rms_enc_ch = np.sqrt(np.mean((y_enc[:n_cmp] - y_ref) ** 2, axis=0))
            rms_ana_ch = np.sqrt(np.mean((y_ana[:n_cmp] - y_ref) ** 2, axis=0))
            rms_phy_ch = np.sqrt(np.mean((y_phy[:n_cmp] - y_ref) ** 2, axis=0))

            traj_rms_enc.append(rms_e)
            traj_rms_ana.append(rms_a)
            traj_rms_phy.append(rms_p)

            windows.append({
                'y_ref': y_ref.copy(),
                'y_enc': y_enc[:n_cmp].copy(),
                'y_ana': y_ana[:n_cmp].copy(),
                'y_phy': y_phy[:n_cmp].copy(),
                'x_traj_enc': x_traj_enc[:n_cmp].copy(),
                'x_traj_ana': x_traj_ana[:n_cmp].copy(),
                'x_traj_phy': x_traj_phy[:n_cmp].copy(),
                'x0_enc': x0_enc.copy(),
                'x0_ana': x0_ana_full.copy(),
                'x0_phy': x0_phy.copy(),
                'rms_enc_ch': rms_enc_ch,
                'rms_ana_ch': rms_ana_ch,
                'rms_phy_ch': rms_phy_ch,
                'rms_enc': rms_e,
                'rms_ana': rms_a,
                'rms_phy': rms_p,
                't_start': int(t_s),
                'n_cmp': n_cmp,
            })

        mean_e = np.mean(traj_rms_enc)
        mean_a = np.mean(traj_rms_ana)
        mean_p = np.mean(traj_rms_phy)
        rms_encoder_all.extend(traj_rms_enc)
        rms_analytical_all.extend(traj_rms_ana)
        rms_physics_all.extend(traj_rms_phy)

        trajectories.append({
            'label': label,
            'skipped': False,
            'starts': starts,
            'windows': windows,
            'mean_rms_enc': mean_e,
            'mean_rms_ana': mean_a,
            'mean_rms_phy': mean_p,
        })

        best = min(mean_e, mean_a, mean_p)
        winner = "Encoder" if best == mean_e else ("Ana" if best == mean_a else "Physics")
        print(f"  {label:<40s}  {len(starts):>4d}  "
              f"{mean_e:10.6f}  {mean_a:10.6f}  {mean_p:10.6f}  {winner:>8s}")

    overall_enc = np.mean(rms_encoder_all)
    overall_ana = np.mean(rms_analytical_all)
    overall_phy = np.mean(rms_physics_all)
    print(f"\n  {'OVERALL':<40s}  {len(rms_encoder_all):>4d}  "
          f"{overall_enc:10.6f}  {overall_ana:10.6f}  {overall_phy:10.6f}")

    if overall_enc < overall_phy:
        pct = (1 - overall_enc / overall_phy) * 100
        print(f"\n  Encoder OUTPERFORMS physics baseline by {pct:.1f}%.")
    else:
        pct = (overall_enc / overall_phy - 1) * 100
        print(f"\n  Encoder is WORSE than physics baseline by {pct:.1f}%.")

    return {
        'overall_enc': overall_enc,
        'overall_ana': overall_ana,
        'overall_phy': overall_phy,
        'trajectories': trajectories,
        'nf': nf,
        'cheat_n': cheat_n,
    }


# =========================================================================
# Worker function (runs in spawned process)
# =========================================================================

def run_encoder_diagnostic(args):
    """Worker: build model with given encoder, train, evaluate. Returns results dict."""
    import time as _time
    encoder_mode, hp, cpus = args
    torch.set_num_threads(cpus)
    t0 = _time.time()

    # Redirect stdout to per-encoder log file (avoids interleaving)
    log_path = os.path.join(save_dir, f'diag_log_{encoder_mode}_{run_id}.txt')
    log_file = open(log_path, 'w', buffering=1)  # line-buffered for tail -f
    old_stdout = sys.stdout
    sys.stdout = log_file

    try:
        seed = SEED if encoder_mode == 'default' else SEED + 1
        np.random.seed(seed)
        torch.manual_seed(seed)

        tag = encoder_mode
        print(f"\n[{tag}] Building model...")
        fit_sys = build_model(hp, encoder_mode=encoder_mode)
        ic_phy = build_physics_only()

        fit_sys.init_model(sys_data=train_data, auto_fit_norm=False)
        if encoder_mode == 'hybrid':
            fit_sys.hfn.to(DTYPE_PT)
        else:
            for net in (fit_sys.encoder, fit_sys.hfn):
                net.to(DTYPE_PT)

        n_params = sum(p.numel() for p in fit_sys.encoder.parameters())
        print(f"[{tag}] Encoder: {n_params} learnable parameters")

        grad_ok = check_gradient_flow(fit_sys, hp, label=tag)
        if not grad_ok:
            return {
                'encoder_mode': encoder_mode,
                'grad_ok': False,
                'train_result': None,
                'rollout_result': None,
                'n_params': n_params,
                'elapsed': _time.time() - t0,
                'log_path': log_path,
            }

        np.random.seed(seed)
        torch.manual_seed(seed)
        train_result = short_training(fit_sys, hp, label=tag)
        rollout_result = multi_window_rollouts(fit_sys, ic_phy, hp, label=tag)

        return {
            'encoder_mode': encoder_mode,
            'grad_ok': grad_ok,
            'train_result': train_result,
            'rollout_result': rollout_result,
            'n_params': n_params,
            'elapsed': _time.time() - t0,
            'log_path': log_path,
        }
    finally:
        sys.stdout = old_stdout
        log_file.close()


# =========================================================================
# Plot functions (per-encoder)
# =========================================================================

CH_LABELS = ['X1 [m]', 'X2 [m]', 'Y [m]']


def plot_training_curves(train_result, save_dir, prefix=''):
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
    ax.set_title(f'Training convergence ({prefix})' if prefix else 'Training convergence')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    ftag = f'_{prefix}' if prefix else ''
    path = os.path.join(save_dir, f'diag_training_curves{ftag}_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_per_trajectory_rollouts(rollout_result, save_dir, prefix=''):
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
                ax.plot(t_plot, win['y_ana'][:, ch], 'C1--', lw=0.8, label='Ana (trained ANN)')
                ax.plot(t_plot, win['y_phy'][:, ch], 'C2:', lw=0.8, label='Physics only')
                ax.grid(True, alpha=0.3)
                if col == 0:
                    ax.set_ylabel(CH_LABELS[ch], fontsize=8)
                if ch == 0:
                    ax.set_title(f't0={win["t_start"]*TS_NEW:.2f}s', fontsize=8)
                    if col == 0:
                        ax.legend(fontsize=6)
                if ch == 2:
                    ax.set_xlabel('Time [s]', fontsize=7)

        ptag = f' ({prefix})' if prefix else ''
        fig.suptitle(f'{label_stem}: enc vs ana vs physics{ptag}', fontsize=10)
        fig.tight_layout()
        ftag = f'_{prefix}' if prefix else ''
        path = os.path.join(save_dir, f'diag_rollout_{label_stem}{ftag}_{run_id}.png')
        fig.savefig(path, dpi=150)
        print(f"  Saved: {os.path.basename(path)}")


def plot_per_channel_bar_chart(rollout_result, hp, save_dir, prefix=''):
    """Plot 3: per-channel RMS bar chart (X1, X2, Y separately)."""
    trajs = rollout_result['trajectories']
    active = [t for t in trajs if not t['skipped']]
    labels = [os.path.splitext(t['label'])[0] for t in active]

    # Compute per-channel means for each trajectory
    means_enc = np.zeros((len(active), 3))
    means_ana = np.zeros((len(active), 3))
    means_phy = np.zeros((len(active), 3))
    for i, t in enumerate(active):
        enc_ch = np.array([w['rms_enc_ch'] for w in t['windows']])
        ana_ch = np.array([w['rms_ana_ch'] for w in t['windows']])
        phy_ch = np.array([w['rms_phy_ch'] for w in t['windows']])
        means_enc[i] = enc_ch.mean(axis=0)
        means_ana[i] = ana_ch.mean(axis=0)
        means_phy[i] = phy_ch.mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    x_pos = np.arange(len(active))
    bar_w = 0.25

    for ch in range(3):
        ax = axes[ch]
        ax.bar(x_pos - bar_w, means_enc[:, ch], bar_w,
               label='Encoder', color='C0')
        ax.bar(x_pos, means_ana[:, ch], bar_w,
               label='Ana (trained ANN)', color='C1')
        ax.bar(x_pos + bar_w, means_phy[:, ch], bar_w,
               label='Physics only', color='C2')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_title(CH_LABELS[ch])
        ax.set_ylabel('Mean RMS [m]')
        ax.grid(True, alpha=0.3, axis='y')
        if ch == 0:
            ax.legend(fontsize=7)

    ptag = f' ({prefix})' if prefix else ''
    fig.suptitle(f'Per-channel RMS: enc vs ana vs physics ({hp["epochs"]} ep){ptag}',
                 fontsize=10)
    fig.tight_layout()
    ftag = f'_{prefix}' if prefix else ''
    path = os.path.join(save_dir, f'diag_per_channel_rms{ftag}_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_error_over_time(rollout_result, save_dir, prefix=''):
    """Plot 4: instantaneous RMS error vs rollout timestep, averaged over all windows."""
    nf = rollout_result['nf']

    # Collect per-timestep squared errors from all windows
    err_enc_list = []
    err_ana_list = []
    err_phy_list = []
    for traj in rollout_result['trajectories']:
        if traj['skipped']:
            continue
        for win in traj['windows']:
            n = win['n_cmp']
            # Pad to nf if shorter (edge windows)
            e_enc = np.full((nf, 3), np.nan)
            e_ana = np.full((nf, 3), np.nan)
            e_phy = np.full((nf, 3), np.nan)
            e_enc[:n] = (win['y_enc'] - win['y_ref']) ** 2
            e_ana[:n] = (win['y_ana'] - win['y_ref']) ** 2
            e_phy[:n] = (win['y_phy'] - win['y_ref']) ** 2
            err_enc_list.append(e_enc)
            err_ana_list.append(e_ana)
            err_phy_list.append(e_phy)

    err_enc = np.array(err_enc_list)  # (n_windows, nf, 3)
    err_ana = np.array(err_ana_list)
    err_phy = np.array(err_phy_list)

    # RMS across windows and channels at each timestep
    rms_enc_t = np.sqrt(np.nanmean(err_enc, axis=(0, 2)))  # (nf,)
    rms_ana_t = np.sqrt(np.nanmean(err_ana, axis=(0, 2)))
    rms_phy_t = np.sqrt(np.nanmean(err_phy, axis=(0, 2)))

    # Also per-channel
    rms_enc_t_ch = np.sqrt(np.nanmean(err_enc, axis=0))  # (nf, 3)
    rms_ana_t_ch = np.sqrt(np.nanmean(err_ana, axis=0))
    rms_phy_t_ch = np.sqrt(np.nanmean(err_phy, axis=0))

    t_axis = np.arange(nf) * TS_NEW

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: overall
    ax = axes[0]
    ax.plot(t_axis, rms_enc_t, 'C0', lw=1, label='Encoder')
    ax.plot(t_axis, rms_ana_t, 'C1', lw=1, label='Ana (trained ANN)')
    ax.plot(t_axis, rms_phy_t, 'C2', lw=1, label='Physics only')
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
        ax.plot(t_axis, rms_phy_t_ch[:, ch], f'C{ch}:', lw=0.8,
                label=f'{CH_LABELS[ch]} phy')
    ax.set_xlabel('Rollout time [s]')
    ax.set_ylabel('RMS error [m]')
    ax.set_title('Per-channel breakdown')
    ax.legend(fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    ftag = f'_{prefix}' if prefix else ''
    path = os.path.join(save_dir, f'diag_error_over_time{ftag}_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_encoder_state_inspection(rollout_result, hp, save_dir, prefix=''):
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

    ptag = f' ({prefix})' if prefix else ''
    fig.suptitle(f'Encoder state inspection: physical (top) + ANN (bottom){ptag}',
                 fontsize=10)
    fig.tight_layout()
    ftag = f'_{prefix}' if prefix else ''
    path = os.path.join(save_dir, f'diag_encoder_states{ftag}_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


# =========================================================================
# Comparison plot functions (side-by-side when running both encoders)
# =========================================================================

MODE_COLORS = {'default': 'C0', 'hybrid': 'C3'}
MODE_LABELS = {'default': 'Default encoder', 'hybrid': 'Hybrid encoder'}


def plot_cmp_training(all_results, save_dir):
    """Comparison: overlay training curves for all encoder modes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for mode, res in all_results.items():
        if not res['grad_ok']:
            continue
        tr = res['train_result']
        c = MODE_COLORS[mode]
        ax.semilogy(tr['epoch_id'], tr['loss_val'], color=c,
                     label=f'{MODE_LABELS[mode]} val')
        ax.semilogy(tr['epoch_id'], tr['loss_train'], color=c,
                     linestyle='--', alpha=0.5, label=f'{MODE_LABELS[mode]} train')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('sim-RMS')
    ax.set_title('Training convergence comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_cmp_training_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_cmp_rms_bar(all_results, hp, save_dir):
    """Comparison: per-channel RMS grouped bars (default enc | hybrid enc | physics)."""
    modes = [m for m in all_results if all_results[m]['grad_ok']]
    if not modes:
        return

    # Use first mode's rollout for trajectory labels and physics baseline
    ref_mode = modes[0]
    trajs_ref = all_results[ref_mode]['rollout_result']['trajectories']
    active_idx = [i for i, t in enumerate(trajs_ref) if not t['skipped']]
    labels = [os.path.splitext(trajs_ref[i]['label'])[0] for i in active_idx]

    n_bars = len(modes) + 1  # +1 for physics
    bar_w = 0.8 / n_bars
    x_pos = np.arange(len(active_idx))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ch in range(3):
        ax = axes[ch]
        for bi, mode in enumerate(modes):
            trajs = all_results[mode]['rollout_result']['trajectories']
            means = np.array([
                np.mean([w['rms_enc_ch'][ch] for w in trajs[i]['windows']])
                for i in active_idx
            ])
            ax.bar(x_pos + bi * bar_w, means, bar_w,
                   label=MODE_LABELS[mode], color=MODE_COLORS[mode])

        # Physics baseline (from first mode — identical across modes)
        phy_means = np.array([
            np.mean([w['rms_phy_ch'][ch] for w in trajs_ref[i]['windows']])
            for i in active_idx
        ])
        ax.bar(x_pos + len(modes) * bar_w, phy_means, bar_w,
               label='Physics only', color='C2')

        ax.set_xticks(x_pos + bar_w * len(modes) / 2)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_title(CH_LABELS[ch])
        ax.set_ylabel('Mean RMS [m]')
        ax.grid(True, alpha=0.3, axis='y')
        if ch == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f'Encoder comparison: per-channel RMS ({hp["epochs"]} epochs)', fontsize=10)
    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_cmp_rms_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


def plot_cmp_error_over_time(all_results, save_dir):
    """Comparison: instantaneous RMS error vs rollout step for each encoder."""
    modes = [m for m in all_results if all_results[m]['grad_ok']]
    if not modes:
        return

    nf = all_results[modes[0]]['rollout_result']['nf']
    t_axis = np.arange(nf) * TS_NEW

    fig, ax = plt.subplots(figsize=(10, 4))

    for mode in modes:
        rollout = all_results[mode]['rollout_result']
        err_list = []
        for traj in rollout['trajectories']:
            if traj['skipped']:
                continue
            for win in traj['windows']:
                n = win['n_cmp']
                e = np.full((nf, 3), np.nan)
                e[:n] = (win['y_enc'] - win['y_ref']) ** 2
                err_list.append(e)
        rms_t = np.sqrt(np.nanmean(np.array(err_list), axis=(0, 2)))
        ax.plot(t_axis, rms_t, color=MODE_COLORS[mode], lw=1,
                label=MODE_LABELS[mode])

    # Physics baseline (from first mode)
    rollout_ref = all_results[modes[0]]['rollout_result']
    phy_err_list = []
    for traj in rollout_ref['trajectories']:
        if traj['skipped']:
            continue
        for win in traj['windows']:
            n = win['n_cmp']
            e = np.full((nf, 3), np.nan)
            e[:n] = (win['y_phy'] - win['y_ref']) ** 2
            phy_err_list.append(e)
    rms_phy_t = np.sqrt(np.nanmean(np.array(phy_err_list), axis=(0, 2)))
    ax.plot(t_axis, rms_phy_t, color='C2', lw=1, linestyle=':', label='Physics only')

    ax.set_xlabel('Rollout time [s]')
    ax.set_ylabel('RMS error [m]')
    ax.set_title('Error growth comparison')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f'diag_cmp_error_growth_{run_id}.png')
    fig.savefig(path, dpi=150)
    print(f"  Saved: {os.path.basename(path)}")


# =========================================================================
# Save all data to .npz
# =========================================================================

def save_diagnostic_npz(all_results, hp, save_dir):
    """Save diagnostic data for all encoder modes to .npz."""
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    save_dict = {
        'hp_json': json.dumps(hp),
        'seed': SEED,
        'NX_PHYS': NX_PHYS,
        'NX_ANN': NX_ANN,
        'nxd': nxd,
        'TS_NEW': TS_NEW,
        'x_mean': x_mean,
        'std_x': std_x,
        'ystd': ystd,
        'y0': y0,
        'encoder_modes': json.dumps(list(all_results.keys())),
    }

    for mode, res in all_results.items():
        pfx = f'{mode}_'
        save_dict[pfx + 'grad_ok'] = res['grad_ok']
        save_dict[pfx + 'n_params'] = res['n_params']

        if not res['grad_ok']:
            continue

        tr = res['train_result']
        save_dict[pfx + 'loss_train'] = tr['loss_train']
        save_dict[pfx + 'loss_val'] = tr['loss_val']
        save_dict[pfx + 'epoch_id'] = tr['epoch_id']
        save_dict[pfx + 'bestfit'] = tr['bestfit']
        save_dict[pfx + 'best_epoch_idx'] = tr['best_epoch_idx']

        ro = res['rollout_result']
        save_dict[pfx + 'overall_rms_enc'] = ro['overall_enc']
        save_dict[pfx + 'overall_rms_ana'] = ro['overall_ana']
        save_dict[pfx + 'overall_rms_phy'] = ro['overall_phy']
        save_dict[pfx + 'nf'] = ro['nf']
        save_dict[pfx + 'n_trajectories'] = len(ro['trajectories'])

        for i, traj in enumerate(ro['trajectories']):
            tp = f'{mode}_traj{i}_'
            save_dict[tp + 'label'] = traj['label']
            save_dict[tp + 'skipped'] = traj['skipped']
            if traj['skipped']:
                continue
            save_dict[tp + 'mean_rms_enc'] = traj['mean_rms_enc']
            save_dict[tp + 'mean_rms_ana'] = traj['mean_rms_ana']
            save_dict[tp + 'mean_rms_phy'] = traj['mean_rms_phy']
            save_dict[tp + 'n_windows'] = len(traj['windows'])
            for j, win in enumerate(traj['windows']):
                wp = f'{mode}_traj{i}_win{j}_'
                for key in ('y_ref', 'y_enc', 'y_ana', 'y_phy',
                            'x0_enc', 'x0_ana', 'x0_phy',
                            'rms_enc_ch', 'rms_ana_ch', 'rms_phy_ch'):
                    save_dict[wp + key] = win[key]
                save_dict[wp + 't_start'] = win['t_start']
                save_dict[wp + 'n_cmp'] = win['n_cmp']

    path = os.path.join(save_dir, f'diag_encoder_results_{run_id}.npz')
    np.savez(path, **save_dict)
    print(f"  Saved: {os.path.basename(path)}")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Encoder diagnostic for gantry SUBNET')
    parser.add_argument('--epochs', type=int, default=DEFAULT_HP['epochs'],
                        help=f'Training epochs (default: {DEFAULT_HP["epochs"]})')
    parser.add_argument('--encoder', choices=['default', 'hybrid', 'both'],
                        default='both', help='Encoder mode (default: both)')
    parser.add_argument('--cpus-per-worker', type=int, default=N_CPUS_PER_WORKER,
                        help=f'Torch threads per worker (default: {N_CPUS_PER_WORKER})')
    args = parser.parse_args()

    hp = DEFAULT_HP.copy()
    hp['epochs'] = args.epochs
    cpus = args.cpus_per_worker

    modes = ['default', 'hybrid'] if args.encoder == 'both' else [args.encoder]

    print(f"\nEncoder diagnostic — modes: {modes}, epochs: {hp['epochs']}, "
          f"cpus/worker: {cpus}")
    print(f"Hyperparameters: {hp}")
    print(f"Save dir: {save_dir}")

    # ── Run diagnostics ────────────────────────────────────────────────
    if len(modes) == 2:
        print(f"\nLaunching 2 workers in parallel...")
        for m in modes:
            print(f"  Log: {os.path.join(save_dir, f'diag_log_{m}_{run_id}.txt')}")
        worker_args = [(m, hp, cpus) for m in modes]
        with multiprocessing.Pool(2) as pool:
            results_list = pool.map(run_encoder_diagnostic, worker_args)
    else:
        print(f"\nRunning single encoder: {modes[0]}")
        print(f"  Log: {os.path.join(save_dir, f'diag_log_{modes[0]}_{run_id}.txt')}")
        results_list = [run_encoder_diagnostic((modes[0], hp, cpus))]

    all_results = {r['encoder_mode']: r for r in results_list}

    # Print per-worker summary
    print(f"\n{'='*70}")
    print("Workers finished")
    print(f"{'='*70}")
    for mode, res in all_results.items():
        elapsed = res.get('elapsed', 0)
        m, s = int(elapsed // 60), int(elapsed % 60)
        if res['grad_ok']:
            best = res['train_result']['bestfit']
            print(f"  [{mode}] {m}m{s:02d}s — best val sim-RMS: {best:.6f}")
        else:
            print(f"  [{mode}] {m}m{s:02d}s — GRADIENT CHECK FAILED")

    # ── Per-encoder plots ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}")

    for mode, res in all_results.items():
        if not res['grad_ok']:
            print(f"  [{mode}] Skipping plots — gradient check failed.")
            continue
        plot_training_curves(res['train_result'], save_dir, prefix=mode)
        plot_per_trajectory_rollouts(res['rollout_result'], save_dir, prefix=mode)
        plot_per_channel_bar_chart(res['rollout_result'], hp, save_dir, prefix=mode)
        plot_error_over_time(res['rollout_result'], save_dir, prefix=mode)
        plot_encoder_state_inspection(res['rollout_result'], hp, save_dir, prefix=mode)

    # ── Comparison plots (if both modes ran) ───────────────────────────
    if len(all_results) >= 2:
        print(f"\nGenerating comparison plots...")
        plot_cmp_training(all_results, save_dir)
        plot_cmp_rms_bar(all_results, hp, save_dir)
        plot_cmp_error_over_time(all_results, save_dir)

    # ── Save data ──────────────────────────────────────────────────────
    save_diagnostic_npz(all_results, hp, save_dir)

    plt.close('all')

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  {'Metric':<30s}", end='')
    for mode in all_results:
        print(f"  {mode:>12s}", end='')
    print(f"  {'physics':>12s}")

    # Overall RMS
    phy_rms = None
    for mode, res in all_results.items():
        if res['grad_ok']:
            phy_rms = res['rollout_result']['overall_phy']
            break

    print(f"  {'Overall RMS':<30s}", end='')
    for mode, res in all_results.items():
        if res['grad_ok']:
            print(f"  {res['rollout_result']['overall_enc']:12.6f}", end='')
        else:
            print(f"  {'GRAD FAIL':>12s}", end='')
    print(f"  {phy_rms:12.6f}" if phy_rms else "")

    # Per-channel
    for ch, lbl in enumerate(['X1', 'X2', 'Y']):
        print(f"  {f'  {lbl} RMS':<30s}", end='')
        for mode, res in all_results.items():
            if res['grad_ok']:
                trajs = res['rollout_result']['trajectories']
                ch_rms = np.mean([
                    np.mean([w['rms_enc_ch'][ch] for w in t['windows']])
                    for t in trajs if not t['skipped']
                ])
                print(f"  {ch_rms:12.6f}", end='')
            else:
                print(f"  {'—':>12s}", end='')
        if phy_rms is not None:
            trajs_ref = next(r['rollout_result']['trajectories']
                             for r in all_results.values() if r['grad_ok'])
            phy_ch = np.mean([
                np.mean([w['rms_phy_ch'][ch] for w in t['windows']])
                for t in trajs_ref if not t['skipped']
            ])
            print(f"  {phy_ch:12.6f}", end='')
        print()

    # Parameters + best epoch
    print(f"  {'Encoder parameters':<30s}", end='')
    for mode, res in all_results.items():
        print(f"  {res['n_params']:>12d}", end='')
    print(f"  {'0':>12s}")

    print(f"  {'Best val sim-RMS':<30s}", end='')
    for mode, res in all_results.items():
        if res['grad_ok']:
            print(f"  {res['train_result']['bestfit']:12.6f}", end='')
        else:
            print(f"  {'—':>12s}", end='')
    print()

    # Winner
    valid = {m: r for m, r in all_results.items() if r['grad_ok']}
    if len(valid) >= 2:
        rms_vals = {m: r['rollout_result']['overall_enc'] for m, r in valid.items()}
        best_mode = min(rms_vals, key=rms_vals.get)
        worst_mode = max(rms_vals, key=rms_vals.get)
        pct = (1 - rms_vals[best_mode] / rms_vals[worst_mode]) * 100
        print(f"\n  WINNER: {best_mode} encoder ({pct:.1f}% lower RMS than {worst_mode})")

    if phy_rms is not None:
        for mode, res in valid.items():
            enc_rms = res['rollout_result']['overall_enc']
            if enc_rms < phy_rms:
                pct = (1 - enc_rms / phy_rms) * 100
                print(f"  {mode} beats physics baseline by {pct:.1f}%")
            else:
                pct = (enc_rms / phy_rms - 1) * 100
                print(f"  {mode} is {pct:.1f}% worse than physics baseline")
