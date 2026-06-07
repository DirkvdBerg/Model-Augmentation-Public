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

Usage:
  conda run -n GraduationProject python diagnose_encoder.py
  conda run -n GraduationProject python diagnose_encoder.py --epochs 20
"""

import os
import sys
import argparse
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
from model_augmentation.systems.gantry_ss import Cd, Dd

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

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

# Hyperparameters for diagnostic (same defaults as gantry_interconnect_dynamic.py)
DEFAULT_HP = dict(
    NX_ANN=3,
    n_nodes_per_layer=128,
    n_hidden_layers=3,
    nf=350,
    batch_size=4000,
    lr=7.6e-4,
    epochs=10,      # short training for diagnostic
)

# =========================================================================
# Data loading (identical to gantry_interconnect_dynamic.py)
# =========================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'Matlab-output', 'identification-trajectories-no-multisine')

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE = 'V1_X_sym_Y_mid_sweep.mat'

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=d['u_total'][::D].astype(DTYPE_NP),
        y=d['q1'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

train_list = [load_traj(f) for f in TRAIN_FILES]
train_data = deepSI.System_data_list(train_list)
val_data = load_traj(VAL_FILE)

print(f'Loaded {len(train_list)} training trajectories, 1 val')

# =========================================================================
# Normalisation (identical to gantry_interconnect_dynamic.py)
# =========================================================================

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])

fs = 1.0 / train_list[0].dt
x_logical_list = []
for t in train_list:
    vel = np.diff(t.y, axis=0) * fs
    vel = np.vstack([vel[:1], vel])
    x_logical_list.append(np.hstack([t.y, vel]))
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

save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet')
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

    Uses the same convention as gantry_interconnect_dynamic.py normalisation:
    positions = [X1, X2, Y] directly from measurements (stage coordinates),
    velocities = [dX1, dX2, dY] from finite differences.

    Returns: (6,) unnormalized physical state.
    """
    pos = y_seq[t_idx]
    if 0 < t_idx < len(y_seq) - 1:
        vel = (y_seq[t_idx + 1] - y_seq[t_idx - 1]) / (2 * dt)
    elif t_idx == 0:
        vel = (y_seq[t_idx + 1] - y_seq[t_idx]) / dt
    else:
        vel = (y_seq[t_idx] - y_seq[t_idx - 1]) / dt
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
    uhist = data_norm.u[t_start - nb:t_start].reshape(1, nb, nu)
    yhist = data_norm.y[t_start - na:t_start].reshape(1, na, ny)
    with torch.no_grad():
        x0 = fit_sys.encoder(
            torch.tensor(uhist, dtype=DTYPE_PT),
            torch.tensor(yhist, dtype=DTYPE_PT),
        )
    return x0.squeeze(0).numpy()


def rollout_from_x0(fit_sys, data_norm, x0_norm, t_start, n_steps):
    """Simulate n_steps from normalized x0. Returns y_hat (n_steps, 3) physical."""
    x = torch.tensor(x0_norm.reshape(1, -1), dtype=DTYPE_PT)
    u_norm = torch.tensor(
        np.ascontiguousarray(data_norm.u[t_start:t_start + n_steps]),
        dtype=DTYPE_PT,
    )
    y_list = []
    with torch.no_grad():
        for t in range(min(n_steps, len(u_norm))):
            y_t, x = fit_sys.hfn(x, u_norm[t:t + 1])
            y_list.append(y_t.squeeze(0).numpy())
    y_hat_norm = np.array(y_list)
    return y_hat_norm * ystd + fit_sys.norm.y0


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

    # Compare encoder weights after training
    w_after = {n: p.clone().detach() for n, p in fit_sys.encoder.named_parameters()}

    print(f"\n  Encoder weight changes after {hp['epochs']} epochs:")
    print(f"  {'Layer':<30s}  {'L2 change':>12s}  {'Relative':>12s}")
    total_change = 0.0
    for name in w_before:
        diff = (w_after[name] - w_before[name]).norm().item()
        rel = diff / (w_before[name].norm().item() + 1e-12)
        total_change += diff
        print(f"  {name:<30s}  {diff:12.6f}  {rel:12.6f}")

    if total_change < 1e-10:
        print("\n  ** FAIL: Encoder weights did not change. Training is not updating it. **")
        return False
    else:
        print(f"\n  PASS: Total weight change = {total_change:.6f}")
        return True


# =========================================================================
# DIAGNOSTIC 3: Multi-window rollouts (encoder vs analytical baseline)
# =========================================================================

def multi_window_rollouts(fit_sys, hp):
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    nf = hp['nf']
    na, nb = fit_sys.na, fit_sys.nb
    cheat_n = max(na, nb)

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC 3: Multi-window rollouts (nf={nf} steps = {nf*TS_NEW:.3f} s)")
    print(f"{'='*70}")
    print(f"  Comparing encoder-init vs analytical baseline (pos + finite-diff vel, ANN=0)")

    # Load best checkpoint
    fit_sys.checkpoint_load_system(name='_best')
    fit_sys.eval()

    val_norm = fit_sys.norm.transform(val_data)

    # Sample windows across ALL trajectories (train + val), like training does
    all_data = train_list + [val_data]
    all_labels = TRAIN_FILES + [VAL_FILE]
    all_norms = [fit_sys.norm.transform(d) for d in all_data]

    rms_encoder_all = []
    rms_analytical_all = []

    print(f"\n  {'Trajectory':<40s}  {'Windows':>7s}  "
          f"{'Enc RMS':>10s}  {'Ana RMS':>10s}  {'Winner':>8s}")

    for traj_idx, (data_raw, data_norm, label) in enumerate(
            zip(all_data, all_norms, all_labels)):

        T = len(data_raw.u)
        if T < cheat_n + nf:
            print(f"  {label:<40s}  SKIPPED (too short)")
            continue

        # Evenly spaced starting points within this trajectory
        usable_start = cheat_n
        usable_end = T - nf
        n_windows = min(10, (usable_end - usable_start) // nf)
        if n_windows < 1:
            n_windows = 1
        starts = np.linspace(usable_start, usable_end, n_windows, dtype=int)

        traj_rms_enc = []
        traj_rms_ana = []

        for t_s in starts:
            # Encoder-initialised rollout
            x0_enc = get_encoder_x0(fit_sys, data_norm, t_s)
            y_enc = rollout_from_x0(fit_sys, data_norm, x0_enc, t_s, nf)

            # Analytical baseline: positions + finite-diff velocities, ANN states = 0
            x0_ana_phys = analytical_x0_at(data_raw.y, t_s, TS_NEW)
            x0_ana_norm = normalize_x(x0_ana_phys)
            x0_ana_full = np.zeros(nxd, dtype=DTYPE_NP)
            x0_ana_full[:NX_PHYS] = x0_ana_norm
            y_ana = rollout_from_x0(fit_sys, data_norm, x0_ana_full, t_s, nf)

            # Reference output
            n_cmp = min(nf, len(data_raw.y) - t_s, len(y_enc))
            y_ref = data_raw.y[t_s:t_s + n_cmp]

            rms_e = np.sqrt(np.mean((y_enc[:n_cmp] - y_ref) ** 2))
            rms_a = np.sqrt(np.mean((y_ana[:n_cmp] - y_ref) ** 2))
            traj_rms_enc.append(rms_e)
            traj_rms_ana.append(rms_a)

        mean_e = np.mean(traj_rms_enc)
        mean_a = np.mean(traj_rms_ana)
        rms_encoder_all.extend(traj_rms_enc)
        rms_analytical_all.extend(traj_rms_ana)

        winner = "Encoder" if mean_e < mean_a else "Baseline"
        print(f"  {label:<40s}  {len(starts):>7d}  "
              f"{mean_e:10.6f}  {mean_a:10.6f}  {winner:>8s}")

    # Overall summary
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

    # ---- Plot: per-trajectory RMS comparison ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(all_labels))
    # Collect per-trajectory means
    traj_means_enc = []
    traj_means_ana = []
    offset = 0
    for data_raw in all_data:
        T = len(data_raw.u)
        if T < cheat_n + nf:
            traj_means_enc.append(0)
            traj_means_ana.append(0)
            continue
        usable_start = cheat_n
        usable_end = T - nf
        n_w = min(10, (usable_end - usable_start) // nf)
        if n_w < 1:
            n_w = 1
        traj_means_enc.append(np.mean(rms_encoder_all[offset:offset + n_w]))
        traj_means_ana.append(np.mean(rms_analytical_all[offset:offset + n_w]))
        offset += n_w

    bar_w = 0.35
    ax.bar(x_pos - bar_w/2, traj_means_enc, bar_w, label='Encoder', color='C0')
    ax.bar(x_pos + bar_w/2, traj_means_ana, bar_w, label='Analytical baseline', color='C1')
    short_labels = [os.path.splitext(f)[0] for f in all_labels]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean RMS over windows')
    ax.set_title(f'Encoder vs analytical baseline ({hp["epochs"]} epochs, nf={nf})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'diag_encoder_vs_baseline.png'), dpi=150)
    print(f"\n  Saved: diag_encoder_vs_baseline.png")

    # ---- Plot: example rollouts from validation trajectory ----
    n_examples = min(6, len(val_data.u) // (cheat_n + nf))
    if n_examples > 0:
        ex_starts = np.linspace(cheat_n, len(val_data.u) - nf, n_examples, dtype=int)
        fig2, axes2 = plt.subplots(n_examples, 3, figsize=(14, 2.5 * n_examples),
                                   sharex='col')
        if n_examples == 1:
            axes2 = axes2.reshape(1, -1)
        ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']

        for row, t_s in enumerate(ex_starts):
            x0_enc = get_encoder_x0(fit_sys, val_norm, t_s)
            y_enc = rollout_from_x0(fit_sys, val_norm, x0_enc, t_s, nf)

            x0_ana_phys = analytical_x0_at(val_data.y, t_s, TS_NEW)
            x0_ana_norm = normalize_x(x0_ana_phys)
            x0_ana_full = np.zeros(nxd, dtype=DTYPE_NP)
            x0_ana_full[:NX_PHYS] = x0_ana_norm
            y_ana = rollout_from_x0(fit_sys, val_norm, x0_ana_full, t_s, nf)

            n_cmp = min(nf, len(val_data.y) - t_s, len(y_enc))
            y_ref = val_data.y[t_s:t_s + n_cmp]
            t_plot = np.arange(n_cmp) * TS_NEW

            for ch in range(3):
                ax = axes2[row, ch]
                ax.plot(t_plot, y_ref[:n_cmp, ch], 'k', lw=1, label='Reference')
                ax.plot(t_plot, y_enc[:n_cmp, ch], 'C0', lw=0.8, label='Encoder')
                ax.plot(t_plot, y_ana[:n_cmp, ch], 'C1--', lw=0.8, label='Analytical')
                ax.grid(True, alpha=0.3)
                if row == 0:
                    ax.set_title(ch_labels[ch])
                    ax.legend(fontsize=6)
                if ch == 0:
                    ax.set_ylabel(f't0={t_s*TS_NEW:.2f}s', fontsize=8)

        fig2.suptitle(f'Validation rollouts: encoder vs analytical baseline', fontsize=11)
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_dir, 'diag_encoder_rollouts.png'), dpi=150)
        print(f"  Saved: diag_encoder_rollouts.png")

    plt.close('all')
    return overall_enc, overall_ana


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
    print(f"Encoder initialised: {sum(p.numel() for p in fit_sys.encoder.parameters())} parameters")

    # Diagnostic 1: gradient flow
    grad_ok = check_gradient_flow(fit_sys, hp)
    if not grad_ok:
        print("\nAborting: no point training if gradients don't reach the encoder.")
        sys.exit(1)

    # Diagnostic 2: short training
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_ok = short_training(fit_sys, hp)

    # Diagnostic 3: multi-window rollouts against baseline
    rms_enc, rms_ana = multi_window_rollouts(fit_sys, hp)

    # Final verdict
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
