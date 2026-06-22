"""
plot_reference_encoder.py
-------------------------
Plot reference trajectories vs best encoder model from encoder_io_data.npz.

Produces:
  1. Output prediction (1-step): y_target vs y_hat (encoder + analytical)
  2. Output prediction (n-step): y_target vs y_hat (encoder + analytical)
  3. State reconstruction: x_target vs x_encoder vs x_analytical (physical units)

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/figures/plot_reference_encoder.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Paths
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))

NPZ_PATH = os.path.join(
    PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder', 'encoder_io_data.npz')
OUT_DIR = os.path.join(SCRIPT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Constants
# =============================================================================

STAGE_UNITS = ['m', 'm', 'm']
PHYS_UNITS = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']


def format_rms(rms_val, unit):
    """Format RMS with SI prefix."""
    if rms_val < 1e-3:
        prefix, scale = ('\u03bc', 1e6)
    elif rms_val < 1.0:
        prefix, scale = ('m', 1e3)
    else:
        prefix, scale = ('', 1.0)
    if unit in ('m/s', 'rad/s') and prefix:
        return f'{rms_val * scale:.2f} {prefix}{unit}'
    if unit in ('m', 'rad') and prefix:
        return f'{rms_val * scale:.2f} {prefix}{unit}'
    return f'{rms_val:.3f} {unit}'


def compute_rms(a, b):
    return np.sqrt(np.mean((a - b)**2, axis=0))


def compute_nrms(a, b):
    rms_err = np.sqrt(np.mean((a - b)**2, axis=0))
    rms_ref = np.sqrt(np.mean(b**2, axis=0))
    return rms_err / (rms_ref + 1e-12)


# =============================================================================
# Plotting
# =============================================================================

def plot_output(y_hat, y_hat_ana, y_target, stage_names, fs, horizon_label, out_path):
    """Time-domain overlay: reference vs encoder vs analytical."""
    T = min(4000, y_hat.shape[0])
    t = np.arange(T) / fs

    nrms_enc = compute_nrms(y_hat[:T], y_target[:T])
    nrms_ana = compute_nrms(y_hat_ana[:T], y_target[:T])

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        rms_enc = compute_rms(y_hat[:T, i:i+1], y_target[:T, i:i+1])[0]
        rms_ana = compute_rms(y_hat_ana[:T, i:i+1], y_target[:T, i:i+1])[0]
        ax.plot(t, y_target[:T, i], 'k-', linewidth=0.8, label='reference')
        ax.plot(t, y_hat[:T, i], 'r--', linewidth=0.8,
                label=f'encoder (NRMS={nrms_enc[i]:.2e}, RMS={format_rms(rms_enc, STAGE_UNITS[i])})')
        ax.plot(t, y_hat_ana[:T, i], 'b:', linewidth=0.8, alpha=0.7,
                label=f'analytical (NRMS={nrms_ana[i]:.2e}, RMS={format_rms(rms_ana, STAGE_UNITS[i])})')
        ax.set_ylabel(f'{stage_names[i]} [{STAGE_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Reference vs encoder: {horizon_label} output prediction', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_states(x_enc, x_ana, x_target, state_names, fs, out_path):
    """State reconstruction: reference vs encoder vs analytical (physical units)."""
    n_states = len(state_names)
    T = min(4000, x_enc.shape[0])
    t = np.arange(T) / fs

    fig, axes = plt.subplots(n_states, 1, figsize=(14, 2.5 * n_states), sharex=True)
    for i, ax in enumerate(axes):
        rms_enc = compute_rms(x_enc[:T, i:i+1], x_target[:T, i:i+1])[0]
        rms_ana = compute_rms(x_ana[:T, i:i+1], x_target[:T, i:i+1])[0]
        nrms_enc = compute_nrms(x_enc[:T, i:i+1], x_target[:T, i:i+1])[0]
        nrms_ana = compute_nrms(x_ana[:T, i:i+1], x_target[:T, i:i+1])[0]

        ax.plot(t, x_target[:T, i], 'k-', linewidth=0.8, label='reference')
        ax.plot(t, x_enc[:T, i], 'r--', linewidth=0.8,
                label=f'encoder (NRMS={nrms_enc:.2e}, RMS={format_rms(rms_enc, PHYS_UNITS[i])})')
        ax.plot(t, x_ana[:T, i], 'b:', linewidth=0.8, alpha=0.7,
                label=f'analytical (NRMS={nrms_ana:.2e}, RMS={format_rms(rms_ana, PHYS_UNITS[i])})')
        ax.set_ylabel(f'{state_names[i]} [{PHYS_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Reference vs encoder: state reconstruction (logical coords)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_output_error(y_hat, y_hat_ana, y_target, stage_names, fs,
                      horizon_label, out_path):
    """Prediction error: encoder and analytical vs reference."""
    T = min(4000, y_hat.shape[0])
    t = np.arange(T) / fs

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
    for i, ax in enumerate(axes):
        enc_err = y_hat[:T, i] - y_target[:T, i]
        ana_err = y_hat_ana[:T, i] - y_target[:T, i]
        rms_enc = np.sqrt(np.mean(enc_err**2))
        rms_ana = np.sqrt(np.mean(ana_err**2))
        ax.plot(t, enc_err, 'r-', linewidth=0.6,
                label=f'encoder (RMS={format_rms(rms_enc, STAGE_UNITS[i])})')
        ax.plot(t, ana_err, 'b-', linewidth=0.6, alpha=0.7,
                label=f'analytical (RMS={format_rms(rms_ana, STAGE_UNITS[i])})')
        ax.set_ylabel(f'{stage_names[i]} error [{STAGE_UNITS[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Output prediction error: {horizon_label}', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print(f'Loading: {NPZ_PATH}')
    d = np.load(NPZ_PATH, allow_pickle=True)

    fs = int(d['fs'])
    n_steps = int(d['n_steps'])
    stage_names = list(d['stage_names'])
    state_names = list(d['state_names'])
    std_x = d['std_x'].flatten()
    x_mean = d['x_mean'].flatten()

    # Denormalize states to physical units
    x_enc_phys = d['x_after_norm'] * std_x + x_mean
    x_ana_phys = d['x_analytical_norm'] * std_x + x_mean
    x_tgt_phys = d['x_target_norm'] * std_x + x_mean

    # --- Plot 1: 1-step output prediction ---
    plot_output(
        d['y_hat_1step'], d['y_hat_ana_1step'], d['y_target_1step'],
        stage_names, fs, '1-step',
        os.path.join(OUT_DIR, 'reference_vs_encoder_output_1step.png'))

    # --- Plot 2: n-step output prediction ---
    plot_output(
        d['y_hat_nstep'], d['y_hat_ana_nstep'], d['y_target_nstep'],
        stage_names, fs, f'{n_steps}-step',
        os.path.join(OUT_DIR, 'reference_vs_encoder_output_nstep.png'))

    # --- Plot 2b: n-step output error (encoder vs analytical) ---
    plot_output_error(
        d['y_hat_nstep'], d['y_hat_ana_nstep'], d['y_target_nstep'],
        stage_names, fs, f'{n_steps}-step',
        os.path.join(OUT_DIR, 'reference_vs_encoder_output_nstep_error.png'))

    # --- Plot 3: state reconstruction ---
    plot_states(
        x_enc_phys, x_ana_phys, x_tgt_phys,
        state_names, fs,
        os.path.join(OUT_DIR, 'reference_vs_encoder_states.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
