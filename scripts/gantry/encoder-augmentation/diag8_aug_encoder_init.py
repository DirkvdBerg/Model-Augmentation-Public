"""
diag8_aug_encoder_init.py
--------------------------
Augmentation encoder initialization diagnostic.

Verifies that linear_encoder_init_aug (with D-055 fix) correctly maps the
input/output history to the AUGMENTED system states at epoch 0.

Ground truth comes from the augmented simulation .mat files (USE_MSD=true):
  - x_logical  (T, 6)  physical states from the gantry+MSD simulation
  - delta_a    (T,)    hidden MSD displacement  [augmented state 1]
  - vdelta_a           estimated as gradient(delta_a)*FS_NEW  [augmented state 2]

This is the correct reference for the augmented encoder: the baseline
baseline_states.npz (no MSD) is NOT used here because it comes from a
different simulation (no hidden mass dynamics).

CHECKS (epoch-0 initialization sanity -- not post-training targets)
------
  C1  All physical NRMS < 1.0          (encoder output in signal range, not predicting mean)
  C2  Max velocity NRMS < 0.5          (W^b gives reasonable velocity estimates)
  C3  Encoder output is finite         (no NaN/Inf -- forward pass health)
  C4  Max position NRMS < 0.2         (position tracking with random-ANN perturbation)

NOTE: Augmented channels (delta_a, vdelta_a) are REPORTED but NOT checked.
W^a is randomly initialized -- NRMS >> 1 is expected before training.
The diagnostic verifies the forward pass is valid, not that W^a tracks the MSD.

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag8_aug_encoder_init.py
"""

import os
import sys
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration -- must match gantry_interconnect_dynamic.py
# =============================================================================

NX_PHYS = 6     # q1, q2, q3, dq1, dq2, dq3
NX_ANN  = 2     # MSD: delta_a, vdelta_a  (D-038)
nu      = 3
ny      = 3

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

MODE = 'multisine'
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', MODE)

# HEURISTIC: Jan's rule na=4*nx+1 for ENCODER_INIT='linear_map'; na_right=1 for y(k) access
na       = 4 * NX_PHYS + 1   # = 25
nb       = na                 # = 25
na_right = 1
nb_right = 1

# Window sizes seen by encoder.forward()
_NB_WIN    = nb + nb_right    # = 26
_NA_WIN    = na + na_right    # = 26
_WIN_START = max(_NB_WIN, _NA_WIN) - 1   # = 25  (first valid timestep index)

STATE_NAMES_PHYS = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']
STATE_NAMES_AUG  = ['delta_a', 'vdelta_a']
PHYS_UNITS       = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s']
AUG_UNITS        = ['m', 'm/s']

# Training and validation file lists (must match gantry_interconnect_dynamic.py)
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

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet',
                       'encoder-augmentation')
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Data loading -- reads x_logical AND delta_a from augmented .mat files
# =============================================================================

def _load_u(d):
    return d['u_total'] if 'u_total' in d else d['u']

def load_mat(filename):
    """Load u, y, x_logical, delta_a from augmented simulation .mat file."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u         = _load_u(d)[::D].astype(DTYPE_NP)
    y         = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)  # (N, 6)
    delta_a   = d['delta_a'][::D].astype(DTYPE_NP)    # (N,)
    # Estimate vdelta_a as backward finite difference, same convention as dq in x_logical
    # HEURISTIC: backward FD velocity -- O(Ts) accurate; consistent with x_logical derivation
    vdelta_a       = np.zeros_like(delta_a)
    vdelta_a[1:]   = (delta_a[1:] - delta_a[:-1]) * FS_NEW
    vdelta_a[0]    = vdelta_a[1]
    x_aug = np.stack([delta_a, vdelta_a], axis=1)  # (N, 2)
    return u, y, x_logical, x_aug

# =============================================================================
# Normalization -- mirrors gantry_interconnect_dynamic.py exactly
# =============================================================================

def compute_normalization(train_data):
    """Compute normalization from augmented training trajectories.

    Uses x_logical from the augmented simulation (not baseline_states.npz).
    Mirrors gantry_interconnect_dynamic.py normalization block.
    """
    u_all = np.concatenate([u for u, _, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _, _ in train_data])
    x_all = np.concatenate([x for _, _, x, _ in train_data])  # (N_total, 6)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)  + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP)        + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP)                        + 1e-8
    y0     = y_all.mean(axis=0).astype(DTYPE_NP)   # empirical, D-017

    return dict(x_mean=x_mean, std_x=std_x, std_u=std_u,
                u_mean=u_mean, ystd=ystd, y0=y0,
                u_all=u_all, y_all=y_all, x_all=x_all)

# =============================================================================
# Analytical baseline (P_inv + finite-diff) -- reference for physical channels
# =============================================================================

def compute_analytical_baseline(y, x_gt_phys):
    """Per-channel NRMS of P_inv+FD estimates vs augmented simulation states."""
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    pos = (P_inv_T @ y.T).T
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0]  = vel[1]
    x_ana = np.hstack([pos, vel])

    err     = x_ana - x_gt_phys
    rms_err = np.sqrt(np.mean(err**2,       axis=0))
    rms_gt  = np.sqrt(np.mean(x_gt_phys**2, axis=0))
    nrms    = rms_err / (rms_gt + 1e-12)
    return nrms, x_ana

# =============================================================================
# Build encoder
# =============================================================================

def build_aug_encoder(norm):
    """Build linear_encoder_init_aug with D-055 fix, using augmented sim stats."""
    import deepSI
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

    # sys_data carries u/y/x for normalize_linear_ss_matrices
    # x is from augmented simulation (x_logical), matching training pipeline
    sys_data = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data.x = norm['x_all']   # (N_total, 6) from augmented simulation

    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data)

    enc = linear_encoder_init_aug(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny,
        na=na, nb=nb,
        nx_aug=NX_ANN,
        n_nodes_per_layer=16,
        n_hidden_layers=2,
        flag_linear_only=False,
        # D-055 convention fix
        u_mean=norm['u_mean'], std_u=norm['std_u'],
        y0=norm['y0'],         ystd=norm['ystd'],
        x_mean=norm['x_mean'], std_x=norm['std_x'],
    ).to(DTYPE_PT)

    return enc

# =============================================================================
# Direct encoder evaluation -- sliding-window batch forward pass
# =============================================================================

def evaluate_encoder_direct(enc, val_u, val_y, norm):
    """Direct sliding-window forward pass.

    Returns x_enc (T, NX_PHYS+NX_ANN):
      columns 0-5 : physical states, de-normalized to physical units
      columns 6-7 : augmented states, in normalized space (no physical unit GT
                    for augmented states in the encoder's internal representation)
    """
    u_mean = norm['u_mean'].flatten()
    std_u  = norm['std_u'].flatten()
    y0     = norm['y0']
    ystd   = norm['ystd']
    x_mean = norm['x_mean'].flatten()
    std_x  = norm['std_x'].flatten()

    u_norm = (val_u - u_mean) / std_u
    y_norm = (val_y - y0)     / ystd

    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    u_batch = torch.tensor(u_wins.copy(), dtype=DTYPE_PT)
    y_batch = torch.tensor(y_wins.copy(), dtype=DTYPE_PT)

    enc.eval()
    with torch.no_grad():
        x_enc_norm = enc(u_batch, y_batch).numpy()   # (T, 8)

    # De-normalize physical channels to physical units
    x_enc = x_enc_norm.copy()
    x_enc[:, :NX_PHYS] = x_enc_norm[:, :NX_PHYS] * std_x + x_mean

    return x_enc

# =============================================================================
# NRMS helper
# =============================================================================

def nrms_per_channel(x_hat, x_gt):
    err     = x_hat - x_gt
    rms_err = np.sqrt(np.mean(err**2,  axis=0))
    rms_gt  = np.sqrt(np.mean(x_gt**2, axis=0))
    return rms_err / (rms_gt + 1e-12)

# =============================================================================
# Plots
# =============================================================================

def _plot_nrms_bar(nrms_phys_enc, nrms_phys_ana, nrms_aug_enc, out_path):
    """Bar chart: encoder init NRMS per channel vs analytical baseline (phys),
    and augmented channel NRMS (no GT comparison for vdelta_a)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: physical channels -- encoder vs analytical
    x = np.arange(NX_PHYS)
    w = 0.35
    ratio = nrms_phys_enc / (nrms_phys_ana + 1e-12)
    ax = axes[0]
    bars_enc = ax.bar(x - w/2, nrms_phys_enc, w,
                      label='Encoder init (theory)', color='tab:blue', alpha=0.85)
    ax.bar(x + w/2, nrms_phys_ana, w,
           label='Analytical (P_inv + FD)', color='tab:orange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES_PHYS)
    ax.set_ylabel('NRMS vs augmented simulation GT')
    ax.set_yscale('log')
    ax.set_title('Physical channels: encoder init vs analytical\n(GT = augmented simulation x_logical)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for i, (b, r) in enumerate(zip(bars_enc, ratio)):
        label = f'{r:.2f}x' if r >= 1 else f'1/{1/r:.1f}x'
        ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.15,
                label, ha='center', va='bottom', fontsize=7.5)

    # Right: augmented channels -- encoder only (GT available for delta_a only)
    ax2 = axes[1]
    labels_aug = [f'{n}\n(GT={u})' for n, u in zip(STATE_NAMES_AUG, AUG_UNITS)]
    colors_aug = ['tab:blue', 'tab:purple']
    bars_aug = ax2.bar(np.arange(NX_ANN), nrms_aug_enc, color=colors_aug, alpha=0.85)
    ax2.set_xticks(np.arange(NX_ANN))
    ax2.set_xticklabels(labels_aug)
    ax2.set_ylabel('NRMS vs GT')
    ax2.set_yscale('log')
    ax2.set_title('Augmented channels: encoder init NRMS\n'
                  '(delta_a GT = mat file; vdelta_a GT = FD estimate)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(1.0, color='k', linewidth=1, linestyle='--', label='NRMS=1 (predict mean)')
    ax2.legend(fontsize=8)
    for b, v in zip(bars_aug, nrms_aug_enc):
        ax2.text(b.get_x() + b.get_width()/2, v * 1.15,
                 f'{v:.2e}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('diag8: Augmentation encoder initialization quality\n'
                 f'(NX_ANN={NX_ANN}, D-055 fix, direct forward pass, val set)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def _plot_time_traces_phys(x_enc_phys, x_ana, x_gt_phys,
                           nrms_enc, nrms_ana, out_path, T_show=500):
    """Physical channels: encoder init vs analytical vs augmented GT."""
    T = min(T_show, len(x_enc_phys))
    t = np.arange(T) * TS_NEW * 1000

    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(13, 2.3 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_gt_phys[:T, i],  'k-',  linewidth=0.9, label='Augmented sim (GT)')
        ax.plot(t, x_enc_phys[:T, i], 'b--', linewidth=0.9,
                label=f'Encoder init  NRMS={nrms_enc[i]:.3e}')
        ax.plot(t, x_ana[:T, i],      'r:',  linewidth=0.9,
                label=f'Analytical    NRMS={nrms_ana[i]:.3e}')
        ax.set_ylabel(f'{STATE_NAMES_PHYS[i]} [{PHYS_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('diag8: Physical channel -- encoder init vs analytical vs augmented simulation GT\n'
                 f'(first {T_show} samples = {T_show * TS_NEW * 1000:.0f} ms)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def _plot_time_traces_aug(x_enc_aug_norm, x_gt_aug, norm, out_path, T_show=500):
    """Augmented channels: encoder output (normalized) vs GT (physical)."""
    T = min(T_show, len(x_enc_aug_norm))
    t = np.arange(T) * TS_NEW * 1000

    # Encoder augmented outputs are in normalized space -- we show them
    # alongside the normalized GT so both are on comparable axes
    x_aug_gt_delta_a  = x_gt_aug[:, 0]   # physical units [m]
    x_aug_gt_vdelta_a = x_gt_aug[:, 1]   # physical units [m/s]

    # Normalize GT for comparison (encoder output is normalized)
    da_mean  = x_aug_gt_delta_a.mean()
    da_std   = x_aug_gt_delta_a.std()   + 1e-8
    vda_mean = x_aug_gt_vdelta_a.mean()
    vda_std  = x_aug_gt_vdelta_a.std()  + 1e-8

    da_gt_norm  = (x_aug_gt_delta_a  - da_mean)  / da_std
    vda_gt_norm = (x_aug_gt_vdelta_a - vda_mean) / vda_std

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, da_gt_norm[:T],              'k-',  linewidth=0.9,
            label=f'delta_a GT (normalized by val std={da_std:.3e} m)')
    ax.plot(t, x_enc_aug_norm[:T, 0],       'b--', linewidth=0.9,
            label='Encoder output (normalized space)')
    ax.set_ylabel('delta_a [normalized]')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Augmented state 1: delta_a (MSD displacement)')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, vda_gt_norm[:T],            'k-',  linewidth=0.9,
             label=f'vdelta_a GT (FD estimate, normalized by std={vda_std:.3e} m/s)')
    ax2.plot(t, x_enc_aug_norm[:T, 1],      'b--', linewidth=0.9,
             label='Encoder output (normalized space)')
    ax2.set_ylabel('vdelta_a [normalized]')
    ax2.set_xlabel('Time [ms]')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_title('Augmented state 2: vdelta_a (MSD velocity, FD estimate)')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('diag8: Augmented channels -- encoder init vs GT\n'
                 f'(normalized axes; first {T_show} samples = {T_show * TS_NEW * 1000:.0f} ms)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('diag8: Augmentation encoder initialization (NX_ANN=2, D-055 fix)')
    print('GT: augmented simulation x_logical + delta_a from .mat files')
    print('=' * 70)
    print(f'  NX_PHYS={NX_PHYS}, NX_ANN={NX_ANN}  (total {NX_PHYS+NX_ANN} states)')
    print(f'  FS_NEW={FS_NEW} Hz  (D={D}, Ts={TS_NEW*1000:.3f} ms)')
    print(f'  na/nb={na}/{nb},  na_right={na_right},  nb_right={nb_right}')
    print(f'  window={_NA_WIN} timesteps  ({_NA_WIN * TS_NEW * 1000:.2f} ms history)')

    # --- Load training data (for normalization) ---
    print(f'\nLoading training data from: {TRAJ_DIR}')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    for i, (fname, data) in enumerate(zip(TRAIN_FILES, train_data)):
        u, y, xl, xa = data
        print(f'  T{i+1}: u={u.shape} y={y.shape} x_logical={xl.shape} x_aug={xa.shape}  ({fname})')

    norm = compute_normalization(train_data)
    print(f'\nNormalization (from augmented sim data):')
    print(f'  y0    = {norm["y0"]}')
    print(f'  ystd  = {norm["ystd"]}')
    print(f'  x_mean= {norm["x_mean"].flatten()}')
    print(f'  std_x = {norm["std_x"].flatten()}')

    # --- Load validation data ---
    print(f'\nLoading val data: {VAL_FILE}')
    val_u, val_y, val_x_phys, val_x_aug = load_mat(VAL_FILE)
    print(f'  val_u={val_u.shape}, val_y={val_y.shape}')
    print(f'  val x_logical={val_x_phys.shape}, val x_aug={val_x_aug.shape}')
    print(f'  delta_a: mean={val_x_aug[:,0].mean():.4e} m  std={val_x_aug[:,0].std():.4e} m')
    print(f'  vdelta_a (FD): mean={val_x_aug[:,1].mean():.4e} m/s  std={val_x_aug[:,1].std():.4e} m/s')

    # Align to encoder window start
    x_gt_phys_aligned = val_x_phys[_WIN_START:]
    x_gt_aug_aligned  = val_x_aug[_WIN_START:]
    val_y_aligned     = val_y[_WIN_START:]

    # --- Analytical baseline (for physical channels) ---
    print(f'\n--- Analytical baseline (P_inv + FD) vs augmented sim GT ---')
    nrms_ana, x_ana = compute_analytical_baseline(val_y, val_x_phys)
    x_ana_aligned   = x_ana[_WIN_START:]
    nrms_ana_aligned = nrms_per_channel(x_ana_aligned, x_gt_phys_aligned)
    print(f'  {"State":<8s}  {"NRMS":>12s}  {"Unit":<5s}')
    print(f'  {"-"*8}  {"-"*12}  {"-"*5}')
    for i, name in enumerate(STATE_NAMES_PHYS):
        print(f'  {name:<8s}  {nrms_ana_aligned[i]:>12.4e}  {PHYS_UNITS[i]:<5s}')

    # --- Build encoder ---
    print(f'\nBuilding linear_encoder_init_aug (NX_ANN={NX_ANN}, D-055 fix)...')
    enc = build_aug_encoder(norm)
    n_params = sum(p.numel() for p in enc.parameters())
    print(f'  Parameters: {n_params}  (Wb: {enc.Wb_psi_y.numel()+enc.Wb_psi_u.numel()}'
          f'  Wa: {enc.Wa_psi_y.numel()+enc.Wa_psi_u.numel()})')
    print(f'  D-055 fix enabled: {enc.fix_enabled}')

    # --- Evaluate at initialization ---
    print(f'\nEvaluating encoder at initialization (direct forward pass)...')
    x_enc = evaluate_encoder_direct(enc, val_u, val_y, norm)
    T = min(len(x_enc), len(x_gt_phys_aligned))
    x_enc              = x_enc[:T]
    x_gt_phys_T        = x_gt_phys_aligned[:T]
    x_gt_aug_T         = x_gt_aug_aligned[:T]
    x_ana_T            = x_ana_aligned[:T]

    x_enc_phys = x_enc[:, :NX_PHYS]   # physical, de-normalized
    x_enc_aug  = x_enc[:, NX_PHYS:]   # augmented, normalized space

    # Physical NRMS
    nrms_phys_enc = nrms_per_channel(x_enc_phys, x_gt_phys_T)
    nrms_ana_T    = nrms_per_channel(x_ana_T, x_gt_phys_T)

    # Augmented NRMS (encoder normalized space vs normalized GT)
    # delta_a: normalize GT to match encoder output scale
    da_gt   = x_gt_aug_T[:, 0]
    vda_gt  = x_gt_aug_T[:, 1]
    da_mean, da_std   = da_gt.mean(),   da_gt.std()   + 1e-8
    vda_mean, vda_std = vda_gt.mean(),  vda_gt.std()  + 1e-8
    da_gt_norm  = (da_gt  - da_mean)  / da_std
    vda_gt_norm = (vda_gt - vda_mean) / vda_std
    x_gt_aug_norm = np.stack([da_gt_norm, vda_gt_norm], axis=1)
    nrms_aug_enc = nrms_per_channel(x_enc_aug, x_gt_aug_norm)

    print(f'\n--- Physical channels: encoder NRMS vs augmented simulation GT ---')
    print(f'  {"State":<8s}  {"Encoder NRMS":>12s}  {"Ana NRMS":>10s}  {"Unit":<5s}  {"Better?":>8s}')
    print(f'  {"-"*8}  {"-"*12}  {"-"*10}  {"-"*5}  {"-"*8}')
    for i, name in enumerate(STATE_NAMES_PHYS):
        better = 'YES' if nrms_phys_enc[i] < nrms_ana_T[i] else 'no'
        print(f'  {name:<8s}  {nrms_phys_enc[i]:>12.4e}  {nrms_ana_T[i]:>10.4e}  '
              f'{PHYS_UNITS[i]:<5s}  {better:>8s}')

    print(f'\n--- Augmented channels: encoder NRMS vs GT (normalized) ---')
    print(f'  Note: encoder outputs are in normalized space; GT normalized by val std')
    print(f'  {"State":<10s}  {"Encoder NRMS":>12s}  {"GT std (phys)":>14s}  {"Unit":<5s}')
    print(f'  {"-"*10}  {"-"*12}  {"-"*14}  {"-"*5}')
    print(f'  {"delta_a":<10s}  {nrms_aug_enc[0]:>12.4e}  {da_std:>14.4e}  {"m":<5s}')
    print(f'  {"vdelta_a":<10s}  {nrms_aug_enc[1]:>12.4e}  {vda_std:>14.4e}  {"m/s":<5s}')
    print(f'  (vdelta_a GT is FD estimate -- treat as informative, not authoritative)')

    # --- Pass/fail checks ---
    # Thresholds are for epoch-0 initialization sanity, not post-training targets.
    # Analytical baseline is shown in the report but NOT used as the check threshold:
    # P_inv is kinematically exact for positions (near-zero NRMS), so any relative
    # threshold would be trivially violated by the ANN perturbation on W^b.
    print('\n' + '=' * 70)
    print('PASS/FAIL CHECKS  (epoch-0 initialization sanity)')
    print('=' * 70)

    checks = {}

    # C1: all physical NRMS < 1.0 -- encoder not worse than predicting the mean
    c1 = bool(np.all(nrms_phys_enc < 1.0))
    checks['C1'] = c1
    print(f'  C1  All physical NRMS < 1.0:               {"PASS" if c1 else "FAIL"}  '
          f'(max={nrms_phys_enc.max():.4e})')

    # C2: velocity NRMS < 0.5 -- W^b gives reasonable velocity estimates at init
    # HEURISTIC: 0.5 is a loose absolute bound; ANN random perturbation is O(0.1-0.3)
    vel_idx = [3, 4, 5]
    c2 = bool(np.all(nrms_phys_enc[vel_idx] < 0.5))
    checks['C2'] = c2
    print(f'  C2  All velocity NRMS < 0.5:               {"PASS" if c2 else "FAIL"}  '
          f'(max vel NRMS={nrms_phys_enc[vel_idx].max():.4e})')

    # C3: encoder output is finite -- forward pass does not produce NaN/Inf
    c3 = bool(np.all(np.isfinite(x_enc)))
    checks['C3'] = c3
    print(f'  C3  Encoder output finite (no NaN/Inf):    {"PASS" if c3 else "FAIL"}  '
          f'(finite={np.sum(np.isfinite(x_enc))}/{x_enc.size})')

    # C4: position NRMS < 0.2 -- positions trackable despite ANN random perturbation
    # HEURISTIC: 0.2 covers typical ANN init noise (~10-15% for default weight scales)
    pos_idx = [0, 1, 2]
    c4 = bool(np.all(nrms_phys_enc[pos_idx] < 0.2))
    checks['C4'] = c4
    print(f'  C4  All position NRMS < 0.2:               {"PASS" if c4 else "FAIL"}  '
          f'(max pos NRMS={nrms_phys_enc[pos_idx].max():.4e})')

    print(f'\n  NOTE: Augmented channels (delta_a, vdelta_a) not checked -- W^a is')
    print(f'        randomly initialized; NRMS >> 1 is expected before training.')
    print(f'        delta_a NRMS={nrms_aug_enc[0]:.4e}  vdelta_a NRMS={nrms_aug_enc[1]:.4e}')

    n_pass = sum(checks.values())
    n_total = len(checks)
    print(f'\n  {n_pass}/{n_total} checks PASSED')
    print('=' * 70)
    if n_pass == n_total:
        print('OVERALL: PASS -- augmented encoder init is valid')
    else:
        print('OVERALL: FAIL -- investigate failing checks above')
        failed = [k for k, v in checks.items() if not v]
        print(f'  Failed: {", ".join(failed)}')

    # --- Save results ---
    result_path = os.path.join(OUT_DIR, 'diag8_results.npz')
    np.savez_compressed(
        result_path,
        x_enc_phys=x_enc_phys, x_gt_phys=x_gt_phys_T,
        x_enc_aug=x_enc_aug,   x_gt_aug=x_gt_aug_T,
        x_ana=x_ana_T,
        nrms_phys_enc=nrms_phys_enc, nrms_phys_ana=nrms_ana_T,
        nrms_aug_enc=nrms_aug_enc,
        state_names_phys=np.array(STATE_NAMES_PHYS),
        state_names_aug=np.array(STATE_NAMES_AUG),
    )
    print(f'\nSaved: {result_path}')

    # --- Plots ---
    _plot_nrms_bar(nrms_phys_enc, nrms_ana_T, nrms_aug_enc,
                   os.path.join(OUT_DIR, 'diag8_nrms_bar.png'))
    _plot_time_traces_phys(x_enc_phys, x_ana_T, x_gt_phys_T,
                           nrms_phys_enc, nrms_ana_T,
                           os.path.join(OUT_DIR, 'diag8_time_traces_phys.png'))
    _plot_time_traces_aug(x_enc_aug, x_gt_aug_T, norm,
                          os.path.join(OUT_DIR, 'diag8_time_traces_aug.png'))


if __name__ == '__main__':
    main()
