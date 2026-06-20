"""
diagnostic_nf_lr.py
-------------------
Pipeline validation: sweep (nf, lr) to check whether the linear encoder
initialization is preserved or destroyed after a few training epochs.

If NO combination preserves the initialization, the pipeline has a bug.
If some combinations work, those are viable hyperparameters.

Uses baseline-v2 data downsampled to 50 Hz (guessed; update FS_NEW after
running the downsampling validation script).

Structure follows step1_baseline_equals_system.py for model building.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/diagnostic_nf_lr.py
"""

import os
import sys
import json
import time
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
FS_NEW = 50       # HEURISTIC: guessed; update after downsampling validation
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

# Fixed hyperparameters (not swept)
HP_FIXED = dict(
    NX_ANN=0,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    batch_size=128,
)

# Diagnostic grid
NF_VALUES = [5, 10, 20, 40, 80]
LR_VALUES = [1e-3, 1e-4, 1e-5, 1e-6]
N_DIAG_EPOCHS = 5

# Data (baseline-v2)
TRAJ_DIR = os.path.join(
    PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine', 'baseline-v2',
)

TRAIN_FILES = [
    'T1_Y_osc.mat',
    'T2_X_sym_Y_sweep.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_sym_Y030.mat',
    'T5_theta_Y_coupling.mat',
    'T6_lissajous_XY.mat',
    'T7_full_MIMO.mat',
    'T8_multi_amp.mat',
    'T9_Y_sweep_repeated.mat',
    'T10_multi_axis_repeated.mat',
]
VAL_FILE = 'V1_osc_Y025.mat'

# Output
OUT_DIR = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'diagnostics',
)
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# Data loading
# =============================================================================

def load_mat(filename):
    """Load u, y, x_logical from .mat file, downsample to FS_NEW."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x_logical


def compute_normalization(train_data):
    """Compute normalization constants from training data."""
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
        ystd=ystd, y0=y0,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )


# =============================================================================
# Build model (same as step1_baseline_equals_system.py)
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
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Diagnostic: nf x lr grid for baseline encoder pipeline validation')
    print('=' * 70)
    print(f'FS_NEW = {FS_NEW} Hz (D = {D})')
    print(f'nf values: {NF_VALUES}')
    print(f'lr values: {LR_VALUES}')
    print(f'Epochs per combination: {N_DIAG_EPOCHS}')
    print(f'Grid size: {len(NF_VALUES)} x {len(LR_VALUES)} = '
          f'{len(NF_VALUES) * len(LR_VALUES)} runs\n')

    # --- Load data ---
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f'Loading data from: {TRAJ_DIR}')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical = load_mat(VAL_FILE)

    for i, (fname, (u, y, x)) in enumerate(zip(TRAIN_FILES, train_data)):
        print(f'  {fname}: u={u.shape}, y={y.shape}')
    print(f'  Val ({VAL_FILE}): u={val_u.shape}, y={val_y.shape}')

    # --- Normalization ---
    norm = compute_normalization(train_data)
    print(f'\nstd_x = {norm["std_x"].flatten()}')
    print(f'std_u = {norm["std_u"].flatten()}')

    # --- Build deepSI System_data ---
    train_list = [
        deepSI.System_data(u=u, y=y, dt=TS_NEW)
        for u, y, _ in train_data
    ]
    train_sys_data = deepSI.System_data_list(train_list)
    val_sys_data = deepSI.System_data(u=val_u, y=val_y, dt=TS_NEW)

    # --- Sweep ---
    results = {}

    print()
    print('=' * 70)
    print(f'{"nf":>5s} | {"lr":>8s} | {"init":>10s} | '
          f'{"ep1":>10s} | {"ep1/init":>8s} | '
          f'{"ep5":>10s} | {"ep5/init":>8s} | {"verdict":<12s} | {"time":>5s}')
    print('-' * 90)

    for nf in NF_VALUES:
        for lr in LR_VALUES:
            np.random.seed(SEED)
            torch.manual_seed(SEED)

            # Fresh model with same linear encoder init
            fit_sys = build_model(HP_FIXED, norm)
            # Pass optimizer_kwargs to init_model, NOT to fit().
            # fit() skips optimizer creation when init_model_done=True,
            # so optimizer_kwargs passed to fit() are silently ignored.
            fit_sys.init_model(
                sys_data=train_sys_data,
                auto_fit_norm=False,
                optimizer_kwargs={'lr': lr},
            )
            fit_sys.hfn.to(DTYPE_PT)

            val_measure = f'{nf}-step-RMS'

            t0 = time.time()
            fit_sys.fit(
                train_sys_data=train_sys_data,
                val_sys_data=val_sys_data,
                batch_size=HP_FIXED['batch_size'],
                epochs=N_DIAG_EPOCHS,
                auto_fit_norm=False,
                loss_kwargs={'nf': nf},
                validation_measure=val_measure,
                verbose=False,
            )
            elapsed = time.time() - t0

            # Loss_val[0] = initial (before training), [1:] = after each epoch
            loss_val = np.array(fit_sys.Loss_val)
            init_loss = float(loss_val[0])
            ep1_loss = float(loss_val[1]) if len(loss_val) > 1 else float('nan')
            ep5_loss = float(loss_val[-1]) if len(loss_val) > 1 else float('nan')

            ratio_ep1 = ep1_loss / init_loss if init_loss > 0 else float('inf')
            ratio_ep5 = ep5_loss / init_loss if init_loss > 0 else float('inf')

            # Verdict
            if ratio_ep1 > 2.0:
                verdict = 'JUMP'
            elif ratio_ep5 < 1.0:
                verdict = 'CONVERGING'
            elif ratio_ep5 < 1.5:
                verdict = 'STABLE'
            else:
                verdict = 'DRIFTING'

            results[(nf, lr)] = {
                'init_loss': init_loss,
                'loss_curve': loss_val.tolist(),
                'ratio_ep1': ratio_ep1,
                'ratio_ep5': ratio_ep5,
                'verdict': verdict,
                'elapsed_s': elapsed,
            }

            print(f'{nf:5d} | {lr:8.0e} | {init_loss:10.4e} | '
                  f'{ep1_loss:10.4e} | {ratio_ep1:8.2f} | '
                  f'{ep5_loss:10.4e} | {ratio_ep5:8.2f} | '
                  f'{verdict:<12s} | {elapsed:5.1f}s')

    # --- Summary ---
    print('\n' + '=' * 70)
    print('SUMMARY')
    print('=' * 70)

    converging = [(k, v) for k, v in results.items() if v['verdict'] == 'CONVERGING']
    stable = [(k, v) for k, v in results.items() if v['verdict'] == 'STABLE']
    jumping = [(k, v) for k, v in results.items() if v['verdict'] == 'JUMP']

    if converging:
        print(f'\nCONVERGING ({len(converging)}):')
        for (nf, lr), v in converging:
            print(f'  nf={nf:3d}, lr={lr:.0e}  '
                  f'init={v["init_loss"]:.4e}  ep5={v["loss_curve"][-1]:.4e}  '
                  f'ratio={v["ratio_ep5"]:.2f}')
    if stable:
        print(f'\nSTABLE ({len(stable)}):')
        for (nf, lr), v in stable:
            print(f'  nf={nf:3d}, lr={lr:.0e}  '
                  f'init={v["init_loss"]:.4e}  ep5={v["loss_curve"][-1]:.4e}  '
                  f'ratio={v["ratio_ep5"]:.2f}')
    if jumping:
        print(f'\nJUMP ({len(jumping)}):')
        for (nf, lr), v in jumping:
            print(f'  nf={nf:3d}, lr={lr:.0e}  '
                  f'init={v["init_loss"]:.4e}  ep1={v["loss_curve"][1]:.4e}  '
                  f'ratio={v["ratio_ep1"]:.2f}')

    if not converging and not stable:
        print('\nNO combination preserved the initialization.')
        print('This indicates a pipeline bug (gradient flow, block connections,')
        print('coordinate transform, or normalization issue).')
    else:
        print(f'\n{len(converging) + len(stable)} of {len(results)} combinations '
              f'preserved the initialization.')

    # --- Save JSON ---
    json_results = {
        'config': {
            'fs_new': FS_NEW, 'na': na, 'nb': nb,
            'nf_values': NF_VALUES, 'lr_values': LR_VALUES,
            'n_epochs': N_DIAG_EPOCHS, 'batch_size': HP_FIXED['batch_size'],
            'train_files': TRAIN_FILES, 'val_file': VAL_FILE,
        },
        'results': {
            f'nf={nf}_lr={lr:.0e}': v for (nf, lr), v in results.items()
        },
    }
    json_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nSaved: {json_path}')

    # --- Heatmap plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (metric, label) in enumerate([
        ('ratio_ep1', 'Loss ratio epoch 1 / init'),
        ('ratio_ep5', 'Loss ratio epoch 5 / init'),
    ]):
        grid = np.zeros((len(NF_VALUES), len(LR_VALUES)))
        for i, nf in enumerate(NF_VALUES):
            for j, lr in enumerate(LR_VALUES):
                grid[i, j] = results[(nf, lr)][metric]

        ax = axes[ax_idx]
        im = ax.imshow(
            grid, cmap='RdYlGn_r', aspect='auto',
            vmin=0.5, vmax=3.0,
        )
        ax.set_xticks(range(len(LR_VALUES)))
        ax.set_xticklabels([f'{lr:.0e}' for lr in LR_VALUES])
        ax.set_yticks(range(len(NF_VALUES)))
        ax.set_yticklabels([str(nf) for nf in NF_VALUES])
        ax.set_xlabel('Learning rate')
        ax.set_ylabel('nf (rollout horizon)')
        ax.set_title(label)

        # Annotate cells
        for i in range(len(NF_VALUES)):
            for j in range(len(LR_VALUES)):
                val = grid[i, j]
                color = 'white' if val > 2.0 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f'Baseline encoder diagnostic (fs={FS_NEW} Hz, {N_DIAG_EPOCHS} epochs)',
        fontsize=13,
    )
    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr_heatmap.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {plot_path}')

    # --- Loss curves plot ---
    fig, axes = plt.subplots(
        len(NF_VALUES), 1, figsize=(10, 3 * len(NF_VALUES)), sharex=True,
    )
    colors = ['tab:red', 'tab:orange', 'tab:blue', 'tab:green']

    for i, nf in enumerate(NF_VALUES):
        ax = axes[i]
        for j, lr in enumerate(LR_VALUES):
            curve = results[(nf, lr)]['loss_curve']
            ax.plot(range(len(curve)), curve, 'o-', color=colors[j],
                    label=f'lr={lr:.0e}', markersize=4, linewidth=1.2)
        ax.set_ylabel(f'nf={nf}\nVal loss')
        ax.set_yscale('log')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Validation step (0 = init, 1..N = after epoch)')
    fig.suptitle(
        f'Loss curves per (nf, lr) (fs={FS_NEW} Hz)',
        fontsize=13,
    )
    fig.tight_layout()
    curve_path = os.path.join(OUT_DIR, 'diagnostic_nf_lr_curves.png')
    fig.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {curve_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
