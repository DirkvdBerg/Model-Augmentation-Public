"""
diagnostic_encoder_init_nx2.py
-------------------------------
Verify linear encoder initialization for gantry_interconnect_dynamic.py
when NX_ANN=2 (augmented states added for MSD dynamics).

Question: does LinearInitEncoderWrapper(nx_ann=2) correctly initialize
physical states x[0:6] against x_logical, while ANN states x[6:8] start ~0?

Uses evaluate_encoder_direct (sliding window, no simulation) to isolate
encoder quality from model propagation — same method as step1 and
diagnostic_nf_lr_400hz.py.

Config mirrors gantry_interconnect_dynamic.py exactly:
  - NX_ANN = 2
  - FS_NEW = 4000 Hz
  - na = 4 * NX_PHYS + 1 = 25  (Jan's rule, linear_map)
  - MSD data: multisine/m50/narrowband/ (MA_FRAC=0.50, narrowband)

Saved: simulations/gantry_subnet/encoder/diag_enc_init_nx2.npz
  x_enc_phys  (T, 6)  encoder physical states at init
  x_enc_ann   (T, 2)  encoder ANN states at init (~0 expected)
  x_gt        (T, 6)  x_logical ground truth
  nrms_init   (6,)    per-channel NRMS at init
  nrms_post   (6,)    per-channel NRMS after short training
  nrms_ana    (6,)    analytical baseline NRMS (P_inv + FD)

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-baseline/diagnostic_encoder_init_nx2.py
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

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
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
# Configuration — mirrors gantry_interconnect_dynamic.py
# =============================================================================

NX_PHYS = 6
NX_ANN  = 2       # matches DEFAULT_HP in gantry_interconnect_dynamic.py
nu, ny  = 3, 3
Y_OP    = None    # LPV self-scheduled

FS_ORIG = 20000
FS_NEW  = 4000    # matches gantry_interconnect_dynamic.py
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32
SEED     = 42

# Encoder history: Jan's rule, linear_map (same as gantry_interconnect_dynamic.py)
na       = 4 * NX_PHYS + 1   # = 25  HEURISTIC: Jan's rule of thumb
nb       = na
na_right = 1
nb_right = 1
_NB_WIN    = nb + nb_right    # 26
_NA_WIN    = na + na_right    # 26
_WIN_START = max(_NB_WIN, _NA_WIN) - 1   # = 25

# Short training to check whether init quality survives output-MSE training
N_TRAIN_EPOCHS = 20
HP = dict(
    NX_ANN             = NX_ANN,
    n_nodes_per_layer  = 16,
    n_hidden_layers    = 2,
    up_sample          = 1,     # HEURISTIC: up_sample=1 validated sufficient at 400 Hz (TS=2.5ms); 4kHz step is 0.25ms, far smaller
    nf                 = 50,     # HEURISTIC: ~2 MSD periods at 157 Hz (25 samp/period @ 4kHz); fast diagnostic
    batch_size         = 256,
    lr                 = 5e-4,
)

# Data: MSD oscillatory multisine (MA_FRAC=0.50, narrowband)
MA_FRAC         = 0.50
MULTISINE_BAND  = 'narrowband'

_msd_dir = os.path.join('multisine', f'm{round(MA_FRAC * 100)}')
if MULTISINE_BAND == 'narrowband':
    _msd_dir = os.path.join(_msd_dir, 'narrowband')
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', _msd_dir)

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
VAL_FILE  = 'V1_osc_Y025.mat'

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
PHYS_UNITS  = ['m',  'm',  'm',  'm/s', 'm/s', 'm/s']


# =============================================================================
# Data loading
# =============================================================================

def load_mat(filename):
    """Load u, y, x_logical, delta_a from .mat file, downsample to FS_NEW."""
    d       = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u       = d['u_total'][::D].astype(DTYPE_NP)
    y       = d['y'][::D].astype(DTYPE_NP)
    x_log   = d['x_logical'][::D].astype(DTYPE_NP)
    delta_a = d['delta_a'][::D].astype(DTYPE_NP) if 'delta_a' in d else None
    return u, y, x_log, delta_a


# =============================================================================
# Normalization
# =============================================================================

def compute_normalization(train_data):
    u_all = np.concatenate([u for u, _, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _, _ in train_data])
    x_all = np.concatenate([x for _, _, x, _ in train_data])

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    return dict(x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
                ystd=ystd, y0=y0, u_all=u_all, y_all=y_all, x_all=x_all)


# =============================================================================
# Analytical baseline
# =============================================================================

def compute_analytical_baseline(y, x_logical):
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    pos     = (P_inv_T @ y.T).T                            # THEORY: q = inv(P^T) y
    vel     = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW               # HEURISTIC: backward FD
    vel[0]  = vel[1]
    x_ana   = np.hstack([pos, vel])
    rms_err = np.sqrt(np.mean((x_ana - x_logical)**2, axis=0))
    rms_gt  = np.sqrt(np.mean(x_logical**2, axis=0))
    return rms_err / (rms_gt + 1e-12), x_ana


# =============================================================================
# Build model — exact copy of gantry_interconnect_dynamic.py:build_model
# =============================================================================

def build_model(norm):
    nxd    = NX_PHYS + NX_ANN
    PHY_IX = np.arange(NX_PHYS)

    x_mean = norm['x_mean'];  std_x  = norm['std_x']
    std_u  = norm['std_u'];   u_mean = norm['u_mean']
    ystd   = norm['ystd'];    y0     = norm['y0']

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np   = Dd.numpy()

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=HP['up_sample'],
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=torch.nn.Tanh,
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

    fit_sys = SSE_Interconnect(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            'n_nodes_per_layer': HP['n_nodes_per_layer'],
            'n_hidden_layers':   HP['n_hidden_layers'],
        },
    )
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # Linear encoder init (Hoekstra 2026 Eq. 16-17)
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    sys_data_with_x   = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_with_x.x = norm['x_all']   # x_logical from MSD data (ground truth states)
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

    phys_encoder = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        flag_linear_only=False,
    )
    fit_sys.encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder,
        nx_ann=NX_ANN,
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd,
        x_mean=x_mean, std_x=std_x,
    ).to(DTYPE_PT)

    return fit_sys


# =============================================================================
# Encoder quality — direct forward pass, no simulation
# =============================================================================

def evaluate_encoder_direct(encoder, val_u, val_y, val_x, norm):
    """Sliding-window encoder forward pass.

    Returns:
      nrms_phys  (NX_PHYS,)  NRMS of physical states vs x_logical
      x_enc_phys (T, NX_PHYS) physical states in physical units
      x_enc_ann  (T, NX_ANN)  ANN states (normalized; ~0 at init)
      x_gt       (T, NX_PHYS) ground truth x_logical aligned to window
    """
    u_mean = norm['u_mean'].flatten();  std_u  = norm['std_u'].flatten()
    y0     = norm['y0'];                ystd   = norm['ystd']
    x_mean = norm['x_mean'].flatten();  std_x  = norm['std_x'].flatten()

    u_norm = (val_u - u_mean) / std_u
    y_norm = (val_y - y0)    / ystd

    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    encoder.eval()
    with torch.no_grad():
        x_enc = encoder(
            torch.tensor(u_wins.copy(), dtype=DTYPE_PT),
            torch.tensor(y_wins.copy(), dtype=DTYPE_PT),
        ).numpy()   # (T, NX_PHYS + NX_ANN)

    x_enc_phys = x_enc[:, :NX_PHYS] * std_x + x_mean   # denormalize
    x_enc_ann  = x_enc[:, NX_PHYS:]                     # normalized (no physical unit)

    x_gt = val_x[_WIN_START:]
    T    = min(len(x_enc_phys), len(x_gt))
    x_enc_phys = x_enc_phys[:T]
    x_enc_ann  = x_enc_ann[:T]
    x_gt       = x_gt[:T]

    rms_err   = np.sqrt(np.mean((x_enc_phys - x_gt)**2, axis=0))
    rms_gt    = np.sqrt(np.mean(x_gt**2, axis=0))
    nrms_phys = rms_err / (rms_gt + 1e-12)

    return nrms_phys, x_enc_phys, x_enc_ann, x_gt


# =============================================================================
# Plotting
# =============================================================================

def plot_init_comparison(x_enc_phys, x_enc_ann, x_ana, x_gt,
                         nrms_enc, nrms_ana, title, out_path):
    """Physical states + ANN states at encoder init."""
    T  = min(2000, len(x_enc_phys))
    t  = np.arange(T) / FS_NEW + _WIN_START / FS_NEW

    # Physical states
    fig, axes = plt.subplots(NX_PHYS + NX_ANN, 1,
                             figsize=(14, 2.2 * (NX_PHYS + NX_ANN)), sharex=True)
    for i in range(NX_PHYS):
        ax = axes[i]
        ax.plot(t, x_gt[:T, i],        'k-',  lw=0.8, label='x_logical (GT)')
        ax.plot(t, x_enc_phys[:T, i],  'r--', lw=0.8,
                label=f'encoder (NRMS={nrms_enc[i]:.2e})')
        ax.plot(t, x_ana[:T, i],       'b:',  lw=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e})')
        ax.set_ylabel(STATE_NAMES[i])
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3)

    # ANN states (should be ~0 at init)
    for j in range(NX_ANN):
        ax = axes[NX_PHYS + j]
        ax.plot(t, x_enc_ann[:T, j], 'g-', lw=0.8)
        rms_ann = float(np.sqrt(np.mean(x_enc_ann[:T, j]**2)))
        ax.set_ylabel(f'x_ann[{j}] (norm)\nRMS={rms_ann:.2e}')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'ANN state {j} — expect ~0 at init', fontsize=8)

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_nrms_bar(nrms_dict, title, out_path):
    x = np.arange(NX_PHYS)
    n = len(nrms_dict)
    w = 0.8 / n
    colors = ['tab:orange', 'tab:red', 'tab:blue']
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (label, nrms) in enumerate(nrms_dict.items()):
        ax.bar(x + (j - n/2 + 0.5) * w, nrms, w,
               label=label, color=colors[j % len(colors)], alpha=0.8)
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


def plot_ann_state_evolution(ann_curves, title, out_path):
    """ANN state RMS per epoch — should start near 0 and grow as MSD learned."""
    epochs = list(range(len(next(iter(ann_curves.values())))))
    fig, axes = plt.subplots(NX_ANN, 1, figsize=(10, 3 * NX_ANN), sharex=True)
    if NX_ANN == 1:
        axes = [axes]
    for j, ax in enumerate(axes):
        ax.plot(epochs, ann_curves[j], 'g-o', markersize=3, linewidth=1)
        ax.set_ylabel(f'x_ann[{j}] RMS (norm)')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'ANN state {j} activation — zero at epoch 0, grows as MSD learned')
    axes[-1].set_xlabel('Epoch')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print('=' * 70)
    print(f'Diagnostic: linear encoder init with NX_ANN={NX_ANN}')
    print(f'  Data:    {TRAJ_DIR}')
    print(f'  FS_NEW:  {FS_NEW} Hz  (D={D}, Ts={TS_NEW*1000:.3f} ms)')
    print(f'  na/nb:   {na}/{nb}  ({na * TS_NEW * 1000:.1f} ms history)')
    print(f'  nf:      {HP["nf"]}  ({HP["nf"] * TS_NEW * 1000:.0f} ms horizon)')
    print(f'  Training: {N_TRAIN_EPOCHS} epochs (init quality check only)')
    print('=' * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f'\nLoading data...')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical, val_delta_a = load_mat(VAL_FILE)

    for i, (fname, (u, y, x, da)) in enumerate(zip(TRAIN_FILES, train_data)):
        da_str = 'yes' if da is not None else 'MISSING'
        print(f'  T{i+1:02d} {fname}: {u.shape[0]} samples, delta_a={da_str}')
    print(f'  Val  {VAL_FILE}: {val_u.shape[0]} samples')

    # ── Normalization ─────────────────────────────────────────────────────────
    norm = compute_normalization(train_data)

    # ── Analytical baseline ───────────────────────────────────────────────────
    nrms_ana, x_analytical = compute_analytical_baseline(val_y, val_x_logical)
    x_ana_win = x_analytical[_WIN_START:]  # align to encoder window

    print(f'\nAnalytical baseline NRMS (P_inv + FD):')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name}: {nrms_ana[i]:.4e}  [{PHYS_UNITS[i]}]')

    # ── Build model ───────────────────────────────────────────────────────────
    print('\nBuilding model (NX_ANN=2, linear_map init)...')
    train_list      = [deepSI.System_data(u=u, y=y, dt=TS_NEW) for u, y, _, _ in train_data]
    train_sys_data  = deepSI.System_data_list(train_list)
    val_sys_data    = deepSI.System_data(u=val_u, y=val_y, dt=TS_NEW)

    fit_sys = build_model(norm)
    fit_sys.init_model(sys_data=train_sys_data, auto_fit_norm=False,
                       optimizer_kwargs={'lr': HP['lr']})
    fit_sys.hfn.to(DTYPE_PT)

    # ── Evaluate BEFORE training ──────────────────────────────────────────────
    print('\n--- Encoder init quality BEFORE training ---')
    nrms_init, x_enc_init, x_ann_init, x_gt = evaluate_encoder_direct(
        fit_sys.encoder, val_u, val_y, val_x_logical, norm)

    print(f'  {"State":<6s}  {"enc NRMS":>12s}  {"ana NRMS":>12s}  {"ratio":>8s}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}  {"-"*8}')
    for i, name in enumerate(STATE_NAMES):
        ratio = nrms_init[i] / (nrms_ana[i] + 1e-12)
        print(f'  {name:<6s}  {nrms_init[i]:>12.4e}  {nrms_ana[i]:>12.4e}  {ratio:>8.2f}x')

    print(f'\n  ANN state RMS at init (normalized, expect ~0):')
    for j in range(NX_ANN):
        rms_j = float(np.sqrt(np.mean(x_ann_init[:, j]**2)))
        print(f'    x_ann[{j}]: {rms_j:.4e}')

    # Pass/fail on init
    max_ratio_init = float(np.max(nrms_init / (nrms_ana + 1e-12)))
    ann_max_rms    = float(np.max([np.sqrt(np.mean(x_ann_init[:, j]**2))
                                   for j in range(NX_ANN)]))
    if max_ratio_init < 2.0 and ann_max_rms < 0.1:
        print(f'\nINIT OK: phys states ≤2x analytical (ratio={max_ratio_init:.2f}x), '
              f'ANN states ≈0 (max RMS={ann_max_rms:.2e})')
    else:
        if max_ratio_init >= 2.0:
            print(f'\nINIT WARN: physical state ratio {max_ratio_init:.2f}x > 2.0 '
                  f'(linear map init may not carry over to NX_ANN=2 correctly)')
        if ann_max_rms >= 0.1:
            print(f'\nINIT WARN: ANN state RMS {ann_max_rms:.2e} ≥ 0.1 '
                  f'(ANN states not starting near zero)')

    # ── Short training — track state quality per epoch ────────────────────────
    print(f'\n--- Training {N_TRAIN_EPOCHS} epochs (nf={HP["nf"]}, lr={HP["lr"]:.0e}) ---')
    hdr = f'{"epoch":>5s} | {"val_loss":>10s}'
    for name in STATE_NAMES:
        hdr += f' | {name:>9s}'
    for j in range(NX_ANN):
        hdr += f' | {"ann"+str(j):>9s}'
    print(hdr)
    print('-' * len(hdr))

    # Epoch 0 row
    row = f'{"init":>5s} | {"":>10s}'
    for i in range(NX_PHYS):
        row += f' | {nrms_init[i]:9.3e}'
    for j in range(NX_ANN):
        row += f' | {float(np.sqrt(np.mean(x_ann_init[:, j]**2))):9.3e}'
    print(row, flush=True)

    ann_rms_curves = {j: [float(np.sqrt(np.mean(x_ann_init[:, j]**2)))]
                      for j in range(NX_ANN)}
    nrms_curves    = {name: [float(nrms_init[i])] for i, name in enumerate(STATE_NAMES)}

    for epoch in range(1, N_TRAIN_EPOCHS + 1):
        fit_sys.fit(
            train_sys_data=train_sys_data, val_sys_data=val_sys_data,
            batch_size=HP['batch_size'], epochs=epoch,
            auto_fit_norm=False, loss_kwargs={'nf': HP['nf']},
            validation_measure=f'{HP["nf"]}-step-RMS', verbose=0,
        )
        ep_loss = float(fit_sys.Loss_val[-1])
        nrms_ep, _, x_ann_ep, _ = evaluate_encoder_direct(
            fit_sys.encoder, val_u, val_y, val_x_logical, norm)

        for i, name in enumerate(STATE_NAMES):
            nrms_curves[name].append(float(nrms_ep[i]))
        for j in range(NX_ANN):
            ann_rms_curves[j].append(float(np.sqrt(np.mean(x_ann_ep[:, j]**2))))

        row = f'{epoch:5d} | {ep_loss:10.4e}'
        for i in range(NX_PHYS):
            row += f' | {nrms_ep[i]:9.3e}'
        for j in range(NX_ANN):
            row += f' | {ann_rms_curves[j][-1]:9.3e}'
        print(row, flush=True)

    # ── Evaluate AFTER training ───────────────────────────────────────────────
    fit_sys.checkpoint_load_system(name='_best')
    nrms_post, x_enc_post, x_ann_post, x_gt_post = evaluate_encoder_direct(
        fit_sys.encoder, val_u, val_y, val_x_logical, norm)

    print(f'\n--- After {N_TRAIN_EPOCHS} epochs ---')
    for i, name in enumerate(STATE_NAMES):
        ratio = nrms_post[i] / (nrms_init[i] + 1e-12)
        print(f'  {name}: init={nrms_init[i]:.3e}  post={nrms_post[i]:.3e}  '
              f'ratio={ratio:.2f}x')

    # ── Save ──────────────────────────────────────────────────────────────────
    T_save = min(len(x_enc_init), len(x_gt), len(x_ana_win))
    npz_path = os.path.join(OUT_DIR, 'diag_enc_init_nx2.npz')
    np.savez_compressed(npz_path,
        x_enc_phys  = x_enc_init[:T_save],
        x_enc_ann   = x_ann_init[:T_save],
        x_enc_post  = x_enc_post[:T_save],
        x_ann_post  = x_ann_post[:T_save],
        x_analytical= x_ana_win[:T_save],
        x_gt        = x_gt[:T_save],
        nrms_init   = nrms_init,
        nrms_post   = nrms_post,
        nrms_ana    = nrms_ana,
        ann_rms_init= np.array([ann_rms_curves[j][0] for j in range(NX_ANN)]),
        ann_rms_post= np.array([ann_rms_curves[j][-1] for j in range(NX_ANN)]),
        state_names = np.array(STATE_NAMES),
        fs          = np.float32(FS_NEW),
        win_start   = np.int32(_WIN_START),
    )
    print(f'\nSaved: {npz_path}')

    json_path = os.path.join(OUT_DIR, 'diag_enc_init_nx2.json')
    with open(json_path, 'w') as f:
        json.dump(dict(
            config=dict(NX_ANN=NX_ANN, FS_NEW=FS_NEW, na=na, nb=nb,
                        MA_FRAC=MA_FRAC, MULTISINE_BAND=MULTISINE_BAND),
            hp=HP, n_train_epochs=N_TRAIN_EPOCHS,
            nrms_init   ={n: float(nrms_init[i])  for i, n in enumerate(STATE_NAMES)},
            nrms_post   ={n: float(nrms_post[i])  for i, n in enumerate(STATE_NAMES)},
            nrms_ana    ={n: float(nrms_ana[i])   for i, n in enumerate(STATE_NAMES)},
            ann_rms_init={j: float(ann_rms_curves[j][0])  for j in range(NX_ANN)},
            ann_rms_post={j: float(ann_rms_curves[j][-1]) for j in range(NX_ANN)},
            nrms_curves =nrms_curves,
            ann_rms_curves={str(j): ann_rms_curves[j] for j in range(NX_ANN)},
        ), f, indent=2)
    print(f'Saved: {json_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_init_comparison(
        x_enc_init, x_ann_init, x_ana_win, x_gt,
        nrms_init, nrms_ana,
        f'Encoder init quality — NX_ANN={NX_ANN}, {MULTISINE_BAND} MSD data',
        os.path.join(OUT_DIR, 'diag_enc_init_nx2_before.png'))

    plot_init_comparison(
        x_enc_post, x_ann_post, x_ana_win, x_gt_post,
        nrms_post, nrms_ana,
        f'After {N_TRAIN_EPOCHS} epochs — NX_ANN={NX_ANN}, {MULTISINE_BAND} MSD data',
        os.path.join(OUT_DIR, 'diag_enc_init_nx2_after.png'))

    plot_nrms_bar(
        {'init': nrms_init, f'after {N_TRAIN_EPOCHS}ep': nrms_post, 'analytical': nrms_ana},
        f'Physical state NRMS — NX_ANN={NX_ANN}',
        os.path.join(OUT_DIR, 'diag_enc_init_nx2_nrms.png'))

    plot_ann_state_evolution(
        ann_rms_curves,
        f'ANN state activation per epoch (NX_ANN={NX_ANN}) — starts ~0, grows as MSD learned',
        os.path.join(OUT_DIR, 'diag_enc_init_nx2_ann_states.png'))


if __name__ == '__main__':
    main()
