"""
plot_encoder_baseline.py
------------------------
Four plots for the encoder baseline verification run (run 67910).

For Plot 1, this script loads the already-saved model and runs the encoder
directly on the validation data (no re-training). The other three plots
use only the saved .npz and .json files.

Data sources (hardcoded to run 67910):
    simulations/gantry_subnet/encoder-baseline/encoder_baseline_model_67910  (model)
    data/gantry/matlab/multisine/baseline-v2/                                 (mat files)
    simulations/gantry_subnet/encoder-baseline/encoder_baseline_data_67910.npz
    simulations/gantry_subnet/encoder-baseline/encoder_baseline_results_67910.json

Plots produced (saved to this directory):
    1. encoder_baseline_state_reconstruction.png
       6 x 2 grid: left column = signal overlap (GT / encoder / analytical) with
       per-state NRMS in the legend; right column = error relative to ground truth.
    2. encoder_baseline_nrms_summary.png
       NRMS bar chart: encoder at init, encoder best checkpoint, analytical baseline.
    3. encoder_baseline_rms_epochs.png
       RMS in physical units per epoch for all 6 states.
    4. encoder_baseline_loss.png
       Training and validation loss per epoch (NaN training gaps shown as line breaks).

Usage (run on the cluster where model and data are stored):
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/figures/plot_encoder_baseline.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.io import loadmat

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))

RUN_ID = '67910'

MODEL_PATH = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder-baseline',
    f'encoder_baseline_model_{RUN_ID}')
TRAJ_DIR = os.path.join(
    PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine', 'baseline-v2')
DATA_DIR = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder-baseline')

NPZ_PATH   = os.path.join(DATA_DIR, f'encoder_baseline_data_{RUN_ID}.npz')
JSON_PATH  = os.path.join(DATA_DIR, f'encoder_baseline_results_{RUN_ID}.json')
CACHE_PATH = os.path.join(DATA_DIR, f'encoder_baseline_enc_best_direct_{RUN_ID}.npz')
OUT_DIR    = SCRIPT_DIR

# =============================================================================
# Constants matching run 67910
# =============================================================================

FS_ORIG  = 20000
FS_NEW   = 400
D        = FS_ORIG // FS_NEW    # downsampling factor = 50
DTYPE_NP = np.float32
DTYPE_PT = torch.float32

NX_PHYS = 6
nu      = 3
ny      = 3

na          = 25
nb          = 25
na_right    = 1
nb_right    = 1
_NB_WIN     = nb + nb_right     # 26
_NA_WIN     = na + na_right     # 26
_WIN_START  = max(_NB_WIN, _NA_WIN) - 1   # 25

TRAIN_FILES = [
    'T1_Y_osc.mat', 'T2_X_sym_Y_sweep.mat', 'T3_X_sym_Y000.mat',
    'T4_X_sym_Y030.mat', 'T5_theta_Y_coupling.mat', 'T6_lissajous_XY.mat',
    'T7_full_MIMO.mat', 'T8_multi_amp.mat', 'T9_Y_sweep_repeated.mat',
    'T10_multi_axis_repeated.mat',
]
VAL_FILE = 'V1_osc_Y025.mat'

# =============================================================================
# Coordinate names and units (logical coordinates)
# State order: q1=X, q2=Theta, q3=Y, q4=dX, q5=dTheta, q6=dY
# =============================================================================

DISPLAY_NAMES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
PHYS_UNITS    = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']

# States shown in the epoch RMS plot (all 6)
OBS_IDX = [0, 1, 2, 3, 4, 5]   # X, Theta, Y, dX, dTheta, dY


# =============================================================================
# Data loading and normalization
# =============================================================================

def load_mat_file(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP) if 'u_total' in d else d['u'][::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x


def compute_normalization(train_data):
    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
    x_all = np.concatenate([x for _, _, x in train_data])
    x_mean = x_all.mean(axis=0).astype(DTYPE_NP)
    std_x  = (x_all.std(axis=0) + 1e-8).astype(DTYPE_NP)
    u_mean = u_all.mean(axis=0).astype(DTYPE_NP)
    std_u  = (u_all.std(axis=0) + 1e-8).astype(DTYPE_NP)
    y0     = y_all.mean(axis=0).astype(DTYPE_NP)
    ystd   = (y_all.std(axis=0) + 1e-8).astype(DTYPE_NP)
    return dict(x_mean=x_mean, std_x=std_x, u_mean=u_mean, std_u=std_u,
                y0=y0, ystd=ystd)


# =============================================================================
# Encoder evaluation (direct, no simulation)
# =============================================================================

def run_encoder_direct(encoder, val_u, val_y, norm):
    """Apply encoder to sliding windows of val I/O data.

    Returns x_enc_phys (T, 6) in physical units.
    """
    u_norm = (val_u - norm['u_mean']) / norm['std_u']
    y_norm = (val_y - norm['y0'])    / norm['ystd']

    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_norm, (_NB_WIN, nu)).reshape(-1, _NB_WIN, nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_norm, (_NA_WIN, ny)).reshape(-1, _NA_WIN, ny)

    u_batch = torch.tensor(u_wins.copy(), dtype=DTYPE_PT)
    y_batch = torch.tensor(y_wins.copy(), dtype=DTYPE_PT)

    encoder.eval()
    with torch.no_grad():
        x_enc_norm = encoder(u_batch, y_batch).numpy()

    return x_enc_norm * norm['std_x'] + norm['x_mean']


# =============================================================================
# Plot 1 -- state reconstruction: overlap (left) + error (right), 6 x 2 grid
# =============================================================================

def plot_state_reconstruction(x_enc, x_ana, x_gt, nrms_enc, nrms_ana, out_path):
    T_show = min(2000, len(x_enc))
    t = np.arange(T_show) / FS_NEW + _WIN_START / FS_NEW

    fig, axes = plt.subplots(NX_PHYS, 2, figsize=(16, 2.5 * NX_PHYS),
                             sharex=True,
                             gridspec_kw={'width_ratios': [2, 1]})

    for i in range(NX_PHYS):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]

        # Left: signal overlap with per-state NRMS in legend
        ax_l.plot(t, x_gt[:T_show, i],  'k-',  linewidth=0.8, label='ground truth')
        ax_l.plot(t, x_enc[:T_show, i], 'r--', linewidth=0.8,
                  label=f'encoder  NRMS={nrms_enc[i]:.2e}')
        ax_l.plot(t, x_ana[:T_show, i], 'b:',  linewidth=0.8,
                  label=f'analytical  NRMS={nrms_ana[i]:.2e}')
        ax_l.set_ylabel(f'{DISPLAY_NAMES[i]} [{PHYS_UNITS[i]}]')
        ax_l.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_l.legend(loc='upper right', fontsize=7)
        ax_l.grid(True, alpha=0.3)

        # Right: error relative to ground truth
        err_enc = x_enc[:T_show, i] - x_gt[:T_show, i]
        err_ana = x_ana[:T_show, i] - x_gt[:T_show, i]
        ax_r.plot(t, err_enc, 'r-', linewidth=0.7, label='encoder error')
        ax_r.plot(t, err_ana, 'b-', linewidth=0.7, label='analytical error')
        ax_r.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax_r.set_ylabel(f'error [{PHYS_UNITS[i]}]')
        ax_r.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax_r.legend(loc='upper right', fontsize=7)
        ax_r.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')
    fig.suptitle(f'State estimation from I/O history, best checkpoint, run {RUN_ID}', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Plot 2 -- NRMS bar chart
# =============================================================================

def plot_nrms_summary(nrms_init, nrms_best, nrms_ana, out_path):
    # Analytical NRMS = 0 exactly for Y (P_inv is perfect): floor for log scale
    nrms_ana_plot = np.maximum(nrms_ana, 1e-10)

    x = np.arange(NX_PHYS)
    w = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, nrms_init,     w, label='encoder, init',            color='tab:orange', alpha=0.85)
    ax.bar(x,     nrms_best,     w, label='encoder, best checkpoint', color='tab:red',    alpha=0.85)
    ax.bar(x + w, nrms_ana_plot, w, label='analytical baseline',      color='tab:blue',   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(f'State estimation NRMS, run {RUN_ID}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Plot 3 -- RMS in physical units vs epoch for X, Y, dX, dY
# =============================================================================

def plot_rms_epochs(state_curves, nrms_init, rms_gt, state_names, cutoff, out_path):
    """Plot per-epoch RMS in physical units for all states.

    state_curves: dict of NRMS per epoch per state (from json)
    nrms_init: array (NX_PHYS,) from nrms_init_direct
    rms_gt: array (NX_PHYS,) RMS of ground truth signal -- used to convert NRMS to RMS
    cutoff: number of epochs to show (same truncation as loss plot)
    """
    epochs = list(range(1, cutoff + 1))
    fig, axes = plt.subplots(len(OBS_IDX), 1, figsize=(10, 2.5 * len(OBS_IDX)), sharex=True)
    for ax, idx in zip(axes, OBS_IDX):
        name = state_names[idx]
        rms_curve    = np.array(state_curves[name])[:cutoff] * rms_gt[idx]
        rms_init_val = nrms_init[idx] * rms_gt[idx]
        ax.plot(epochs, rms_curve, 'b-o', markersize=3, linewidth=1,
                label='per-epoch RMS')
        ax.axhline(rms_init_val, color='orange', linestyle='--', linewidth=1.2,
                   label='init level')
        ax.set_ylabel(f'{DISPLAY_NAMES[idx]} RMS [{PHYS_UNITS[idx]}]')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Epoch')
    fig.suptitle(f'State estimation RMS vs MATLAB ground truth, run {RUN_ID}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Plot 4 -- training and validation loss per epoch
# =============================================================================

def plot_loss(loss_train, loss_val, cutoff, out_path):
    loss_train_arr = np.array(loss_train, dtype=float)
    loss_val_arr   = np.array(loss_val,   dtype=float)

    epochs = list(range(1, cutoff + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    # NaN entries in loss_train create line breaks automatically
    ax.plot(epochs, loss_train_arr[:cutoff], 'r-o', markersize=3, linewidth=1.2, label='training loss')
    ax.plot(epochs, loss_val_arr[:cutoff],   'k-o', markersize=3, linewidth=1.2, label='validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Training and validation loss, run {RUN_ID}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    # --- Load npz (needed for analytical baseline and ground truth) ---
    print(f'Loading NPZ:  {NPZ_PATH}')
    npz = np.load(NPZ_PATH, allow_pickle=True)

    # --- Encoder: load from cache if available, otherwise run and save ---
    if os.path.exists(CACHE_PATH):
        print(f'Loading cached encoder output: {CACHE_PATH}')
        cache = np.load(CACHE_PATH)
        x_enc_best = cache['x_enc_best_direct']
        nrms_best  = cache['nrms_best_direct']
    else:
        print(f'Cache not found. Loading model: {MODEL_PATH}')
        fit_sys = torch.load(MODEL_PATH, weights_only=False, map_location='cpu')
        encoder = fit_sys.encoder

        print(f'Loading {len(TRAIN_FILES)} training files for normalization...')
        train_data = [load_mat_file(f) for f in TRAIN_FILES]
        norm = compute_normalization(train_data)

        print(f'Loading validation file: {VAL_FILE}')
        val_u, val_y, val_x_logical = load_mat_file(VAL_FILE)

        print('Running encoder on validation data...')
        x_enc_best = run_encoder_direct(encoder, val_u, val_y, norm)

        T            = len(x_enc_best)
        x_gt_direct  = val_x_logical[_WIN_START:_WIN_START + T]
        rms_err      = np.sqrt(np.mean((x_enc_best - x_gt_direct) ** 2, axis=0))
        rms_gt_cache = np.sqrt(np.mean(x_gt_direct ** 2, axis=0))
        nrms_best    = rms_err / (rms_gt_cache + 1e-12)

        np.savez_compressed(CACHE_PATH,
                            x_enc_best_direct=x_enc_best,
                            nrms_best_direct=nrms_best)
        print(f'Saved cache: {CACHE_PATH}')

    T = len(x_enc_best)

    # Ground truth and analytical aligned to the same window
    x_gt_direct  = npz['x_gt'][_WIN_START:_WIN_START + T]
    x_ana_full   = npz['x_analytical']
    x_ana_direct = x_ana_full[_WIN_START:_WIN_START + T]

    # RMS of ground truth used to convert NRMS -> RMS in physical units (Plot 3)
    rms_gt = np.sqrt(np.mean(x_gt_direct ** 2, axis=0))

    # --- Load json for init NRMS, analytical NRMS, and training curves ---
    print(f'Loading JSON: {JSON_PATH}')
    with open(JSON_PATH) as f:
        js = json.load(f)
    state_names = list(npz['state_names'])

    nrms_init    = np.array([js['nrms_init_direct'][n] for n in state_names])
    nrms_ana     = np.array([js['nrms_analytical'][n]  for n in state_names])
    loss_train   = js['loss_train']
    loss_val     = js['loss_val']
    state_curves = js['state_curves']

    # Plateau cutoff: pipeline records running-best, so curves flatline once
    # training stops improving. Find first repeated val_loss entry and cut
    # to plateau_start + 2 so the onset is visible. Applied to both loss and
    # state curves so both plots have consistent x-axis range.
    loss_val_arr = np.array(loss_val, dtype=float)
    cutoff = len(loss_val_arr)
    for i in range(1, len(loss_val_arr)):
        if loss_val_arr[i] == loss_val_arr[i - 1]:
            cutoff = min(i + 2, len(loss_val_arr))
            break

    # --- Plots ---
    plot_state_reconstruction(
        x_enc_best, x_ana_direct, x_gt_direct, nrms_best, nrms_ana,
        os.path.join(OUT_DIR, 'encoder_baseline_state_reconstruction.png'))

    plot_nrms_summary(
        nrms_init, nrms_best, nrms_ana,
        os.path.join(OUT_DIR, 'encoder_baseline_nrms_summary.png'))

    plot_rms_epochs(
        state_curves, nrms_init, rms_gt, state_names, cutoff,
        os.path.join(OUT_DIR, 'encoder_baseline_rms_epochs.png'))

    plot_loss(
        loss_train, loss_val, cutoff,
        os.path.join(OUT_DIR, 'encoder_baseline_loss.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
