"""
verify_msd_visibility.py
-------------------------
Checks whether the hidden MSD resonance (fa=150 Hz) is visible in the
training data, and whether naive decimation from 20 kHz to 1 kHz destroys it.

Five diagnostics per trajectory:
  1. delta_a amplitude (max, RMS) at 20 kHz
  2. PSD of Y-channel at 20 kHz with 150 Hz marker + delta_a spectrum
  3. PSD of Y-channel at 1 kHz: naive (y[::20]) vs proper (scipy.signal.decimate)
  4. Time-domain Y residual at 20 kHz: measured y minus physics-only rollout
  5. Time-domain Y residual at 1 kHz:  measured y minus physics-only rollout

Comparing 4 vs 5 isolates whether the MSD is inherently tiny on (X1,X2,Y)
or whether the decimation destroys the signal.

Parallelized with multiprocessing.Pool — one process per trajectory.

Outputs (saved to simulations/gantry_subnet/msd_verification/):
  - msd_visibility_<traj>_{rid}.png : 6-panel figure per trajectory
  - msd_summary_{rid}.png           : bar chart across all trajectories
  - msd_results_{rid}.npz           : all numerical results

Usage:
  conda run -n GraduationProject python verify_msd_visibility.py
"""

import os
import sys
import numpy as np
import torch
import multiprocessing
from functools import partial
from datetime import datetime
from scipy.io import loadmat
from scipy.signal import decimate, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================================================================
# Configuration (matches gantry_interconnect_dynamic.py)
# =========================================================================

# Toggle between data sources: 'trajectories' or 'multisine'
MODE = 'multisine'

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None
SEED = 42

FS_ORIG  = 20000
FS_NEW   = 4000
D        = FS_ORIG // FS_NEW   # = 5
TS_NEW   = 1.0 / FS_NEW
TS_ORIG  = 1.0 / FS_ORIG

# MSD parameters (from generate_trajectory_data_without_multisine.m lines 39-46)
# HEURISTIC: these are the values used in the data generation script
FA_MSD   = 150    # [Hz] MSD natural frequency
MA_FRAC  = 0.10   # fraction of mh
MH_TOTAL = 10.1   # [kg]

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

# HEURISTIC: nperseg chosen for ~1 Hz resolution at both sample rates
NPERSEG_20K = 16384   # ~1.2 Hz resolution at 20 kHz
NPERSEG_1K  = 1024    # ~1.0 Hz resolution at 1 kHz

TRAJ_FILES = [
    'T1_Y_sweep_conservative.mat',
    'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',
    'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',
    'T8_X_sym_anti_Y_sweep.mat',
    'V1_X_sym_Y_mid_sweep.mat',
    'E1_X_sym_anti_Y_low_offset_sweep.mat',
]

MULTISINE_FILES = [
    'T1_Y_sweep_conservative.mat',
    'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',
    'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',
    'T8_X_sym_anti_Y_sweep.mat',
    'V1_X_sym_Y_mid_sweep.mat',
    'E1_X_sym_anti_Y_low_offset_sweep.mat',
]

# Select files and training subset based on mode
ALL_FILES   = MULTISINE_FILES if MODE == 'multisine' else TRAJ_FILES
TRAIN_FILES = ALL_FILES[:8]

# Subdirectory under data/gantry/matlab/
DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'


# =========================================================================
# Data loading helpers
# =========================================================================

def _load_u(d):
    """Load plant input force from .mat dict.

    Multisine data saves as 'u_total'; trajectory data saves as 'u'.
    """
    if 'u_total' in d:
        return d['u_total']
    return d['u']


# =========================================================================
# Normalization computation (called once in main process)
# =========================================================================

def compute_normalization(traj_dir):
    """Compute normalization stats at 1 kHz and 20 kHz from training data.

    Returns a dict of numpy arrays (all picklable for multiprocessing).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from model_augmentation.systems.gantry_ss import Cd, Dd, P

    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    Cd_np = Cd.numpy()
    Dd_np = Dd.numpy()

    # --- 1 kHz stats ---
    train_u_list, train_y_list = [], []
    for f in TRAIN_FILES:
        d = loadmat(os.path.join(traj_dir, f), squeeze_me=True)
        train_u_list.append(_load_u(d)[::D].astype(DTYPE_NP))
        train_y_list.append(d['y'][::D].astype(DTYPE_NP))

    u_all = np.concatenate(train_u_list)
    y_all = np.concatenate(train_y_list)

    x_logical_list = []
    for y_tr in train_y_list:
        pos = (P_inv_T @ y_tr.T).T
        vel = np.diff(pos, axis=0) * FS_NEW
        vel = np.vstack([vel[:1], vel])
        x_logical_list.append(np.hstack([pos, vel]))
    x_all = np.concatenate(x_logical_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = (Cd_np @ x_mean.flatten()).astype(DTYPE_NP)
    Cd_norm = Cd_np * std_x.flatten()[None, :] / ystd[:, None]

    # --- 20 kHz stats ---
    train_u_20k, train_y_20k = [], []
    for f in TRAIN_FILES:
        d = loadmat(os.path.join(traj_dir, f), squeeze_me=True)
        train_u_20k.append(_load_u(d).astype(DTYPE_NP))
        train_y_20k.append(d['y'].astype(DTYPE_NP))

    u_all_20k = np.concatenate(train_u_20k)
    y_all_20k = np.concatenate(train_y_20k)

    x_logical_20k = []
    for y_tr in train_y_20k:
        pos = (P_inv_T @ y_tr.T).T
        vel = np.diff(pos, axis=0) * FS_ORIG
        vel = np.vstack([vel[:1], vel])
        x_logical_20k.append(np.hstack([pos, vel]))
    x_all_20k = np.concatenate(x_logical_20k)

    x_mean_20k = x_all_20k.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x_20k  = x_all_20k.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u_20k  = u_all_20k.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean_20k = u_all_20k.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd_20k   = y_all_20k.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0_20k     = (Cd_np @ x_mean_20k.flatten()).astype(DTYPE_NP)
    Cd_norm_20k = Cd_np * std_x_20k.flatten()[None, :] / ystd_20k[:, None]

    return {
        'P_inv_T': P_inv_T, 'Cd_np': Cd_np, 'Dd_np': Dd_np,
        # 1 kHz
        'x_mean': x_mean, 'std_x': std_x, 'u_mean': u_mean, 'std_u': std_u,
        'ystd': ystd, 'y0': y0, 'Cd_norm': Cd_norm,
        # 20 kHz
        'x_mean_20k': x_mean_20k, 'std_x_20k': std_x_20k,
        'u_mean_20k': u_mean_20k, 'std_u_20k': std_u_20k,
        'ystd_20k': ystd_20k, 'y0_20k': y0_20k, 'Cd_norm_20k': Cd_norm_20k,
    }


# =========================================================================
# Physics model builders
# =========================================================================

def _build_physics_ic(norm, rate):
    """Build a physics-only interconnect for the given rate ('1k' or '20k')."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from model_augmentation.fit_systems.interconnect import Interconnect
    from model_augmentation.fit_systems.blocks import (
        Gantry_State_Block, Linear_Output_Block,
    )

    if rate == '1k':
        std_x_b, std_u_b = norm['std_x'], norm['std_u']
        x_mean_b, u_mean_b = norm['x_mean'], norm['u_mean']
        Cd_norm_b, Ts = norm['Cd_norm'], TS_NEW
    else:
        std_x_b, std_u_b = norm['std_x_20k'], norm['std_u_20k']
        x_mean_b, u_mean_b = norm['x_mean_20k'], norm['u_mean_20k']
        Cd_norm_b, Ts = norm['Cd_norm_20k'], TS_ORIG

    ic = Interconnect(NX_PHYS, nu, ny, debugging=False)
    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x_b, std_u=std_u_b,
        x_mean=x_mean_b, u_mean=u_mean_b, Ts=Ts,
    ).to(DTYPE_PT)
    out_block = Linear_Output_Block(C=Cd_norm_b, D=norm['Dd_np'])
    ic.add_block(phy_block)
    ic.add_block(out_block)
    ic.connect_signals("x", phy_block)
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp")
    ic.connect_signals("x", out_block)
    ic.connect_block_signals(out_block, ["u"], ["y"])
    return ic


def analytical_x0(y_seq, t_idx, dt, x_mean_b, std_x_b, P_inv_T):
    """Compute normalized physical state at time index t_idx."""
    pos = P_inv_T @ y_seq[t_idx]
    if 0 < t_idx < len(y_seq) - 1:
        vel = P_inv_T @ ((y_seq[t_idx + 1] - y_seq[t_idx - 1]) / (2 * dt))
    elif t_idx == 0:
        vel = P_inv_T @ ((y_seq[t_idx + 1] - y_seq[t_idx]) / dt)
    else:
        vel = P_inv_T @ ((y_seq[t_idx] - y_seq[t_idx - 1]) / dt)
    x_phys = np.concatenate([pos, vel])
    return ((x_phys.reshape(NX_PHYS, 1) - x_mean_b) / std_x_b).flatten()


def rollout_physics(ic, y_data, u_data, dt, x_mean_b, std_x_b, u_mean_b,
                    std_u_b, ystd_b, y0_b, P_inv_T):
    """Full rollout of physics-only model from analytical x0.

    Returns y_hat (N, 3) in physical units.
    """
    u_norm = (u_data - u_mean_b.flatten()) / std_u_b.flatten()

    x0_norm = analytical_x0(y_data, 0, dt, x_mean_b, std_x_b, P_inv_T)
    x = torch.tensor(x0_norm.reshape(1, -1), dtype=DTYPE_PT)
    u_t = torch.tensor(u_norm, dtype=DTYPE_PT)

    y_list = []
    with torch.no_grad():
        for t in range(len(u_t)):
            y_t, x = ic(x, u_t[t:t + 1])
            y_list.append(y_t.squeeze(0).numpy())

    y_hat_norm = np.array(y_list)
    return y_hat_norm * ystd_b + y0_b


# =========================================================================
# Per-trajectory worker (runs in its own process)
# =========================================================================

def process_trajectory(fname, norm, traj_dir, save_dir, run_id):
    """Process one trajectory: all 5 diagnostics + 6-panel plot.

    Returns results dict.
    """
    label = os.path.splitext(fname)[0]
    P_inv_T = norm['P_inv_T']

    # Build physics models locally (lightweight, avoids pickling nn.Modules)
    ic_1k  = _build_physics_ic(norm, '1k')
    ic_20k = _build_physics_ic(norm, '20k')

    d = loadmat(os.path.join(traj_dir, fname), squeeze_me=True)

    # Raw 20 kHz data (u_total for multisine, u for trajectories)
    u_20k = _load_u(d).astype(np.float64)
    y_20k = d['y'].astype(np.float64)

    has_delta_a = 'delta_a' in d
    if has_delta_a:
        delta_a_20k = d['delta_a'].astype(np.float64).flatten()
    else:
        delta_a_20k = np.zeros(len(y_20k))

    N_20k = len(y_20k)
    t_20k = np.arange(N_20k) / FS_ORIG

    # ------------------------------------------------------------------
    # Diagnostic 1: delta_a amplitude
    # ------------------------------------------------------------------
    da_max = np.max(np.abs(delta_a_20k))
    da_rms = np.sqrt(np.mean(delta_a_20k ** 2))

    # ------------------------------------------------------------------
    # Diagnostic 2: PSD at 20 kHz (Y-channel + delta_a)
    # ------------------------------------------------------------------
    f_psd_20k, psd_y_20k = welch(y_20k[:, 2], fs=FS_ORIG, nperseg=NPERSEG_20K)
    if has_delta_a:
        _, psd_da_20k = welch(delta_a_20k, fs=FS_ORIG, nperseg=NPERSEG_20K)
    else:
        psd_da_20k = np.zeros_like(psd_y_20k)

    # ------------------------------------------------------------------
    # Diagnostic 3: Naive vs proper decimation at 1 kHz
    # ------------------------------------------------------------------
    y_naive = y_20k[::D, 2]
    y_proper = decimate(y_20k[:, 2], D, ftype='fir', zero_phase=True)

    f_1k_n, psd_naive  = welch(y_naive,  fs=FS_NEW, nperseg=NPERSEG_1K)
    f_1k_p, psd_proper = welch(y_proper, fs=FS_NEW, nperseg=NPERSEG_1K)

    # ------------------------------------------------------------------
    # Diagnostic 4: Time-domain Y residual at 20 kHz
    # ------------------------------------------------------------------
    y_20k_f32 = y_20k.astype(DTYPE_NP)
    u_20k_f32 = u_20k.astype(DTYPE_NP)

    y_phy_20k = rollout_physics(
        ic_20k, y_20k_f32, u_20k_f32, TS_ORIG,
        norm['x_mean_20k'], norm['std_x_20k'],
        norm['u_mean_20k'], norm['std_u_20k'],
        norm['ystd_20k'], norm['y0_20k'], P_inv_T)
    n_cmp_20k = min(N_20k, len(y_phy_20k))

    resid_y_20k = y_20k_f32[:n_cmp_20k, 2] - y_phy_20k[:n_cmp_20k, 2]
    resid_rms_20k = float(np.sqrt(np.mean(resid_y_20k ** 2)))
    resid_max_20k = float(np.max(np.abs(resid_y_20k)))
    resid_all_20k = y_20k_f32[:n_cmp_20k] - y_phy_20k[:n_cmp_20k]
    resid_rms_ch_20k = np.sqrt(np.mean(resid_all_20k ** 2, axis=0))

    f_res_20k, psd_res_20k = welch(resid_y_20k, fs=FS_ORIG, nperseg=NPERSEG_20K)

    # ------------------------------------------------------------------
    # Diagnostic 5: Time-domain Y residual at 1 kHz
    # ------------------------------------------------------------------
    y_1k = y_20k[::D].astype(DTYPE_NP)
    u_1k = u_20k[::D].astype(DTYPE_NP)  # u_20k already loaded via _load_u
    N_1k = len(y_1k)
    t_1k = np.arange(N_1k) / FS_NEW

    y_phy_1k = rollout_physics(
        ic_1k, y_1k, u_1k, TS_NEW,
        norm['x_mean'], norm['std_x'],
        norm['u_mean'], norm['std_u'],
        norm['ystd'], norm['y0'], P_inv_T)
    n_cmp_1k = min(N_1k, len(y_phy_1k))

    resid_y_1k = y_1k[:n_cmp_1k, 2] - y_phy_1k[:n_cmp_1k, 2]
    resid_rms_1k = float(np.sqrt(np.mean(resid_y_1k ** 2)))
    resid_max_1k = float(np.max(np.abs(resid_y_1k)))
    resid_all_1k = y_1k[:n_cmp_1k] - y_phy_1k[:n_cmp_1k]
    resid_rms_ch_1k = np.sqrt(np.mean(resid_all_1k ** 2, axis=0))

    f_res_1k, psd_res_1k = welch(resid_y_1k, fs=FS_NEW, nperseg=NPERSEG_1K)

    ratio = resid_rms_1k / resid_rms_20k if resid_rms_20k > 0 else float('inf')

    # ------------------------------------------------------------------
    # Spectral summary (for text output)
    # ------------------------------------------------------------------
    def psd_at_freq(f_arr, psd_arr, target_hz):
        """PSD value at the frequency bin closest to target_hz."""
        idx = np.argmin(np.abs(f_arr - target_hz))
        return float(psd_arr[idx])

    def band_power(f_arr, psd_arr, f_lo, f_hi):
        """Integrated power in [f_lo, f_hi] band (trapezoidal)."""
        mask = (f_arr >= f_lo) & (f_arr <= f_hi)
        if np.sum(mask) < 2:
            return 0.0
        return float(np.trapz(psd_arr[mask], f_arr[mask]))

    def peak_in_band(f_arr, psd_arr, f_lo, f_hi):
        """Peak frequency and PSD value within [f_lo, f_hi]."""
        mask = (f_arr >= f_lo) & (f_arr <= f_hi)
        if np.sum(mask) == 0:
            return 0.0, 0.0
        f_band = f_arr[mask]
        p_band = psd_arr[mask]
        idx = np.argmax(p_band)
        return float(f_band[idx]), float(p_band[idx])

    # Point samples at key frequencies
    freq_targets = [50, 100, FA_MSD, 200, 300, 400, 450]

    spectral = {}
    for ft in freq_targets:
        if ft <= FS_NEW / 2:
            spectral[f'psd_naive_{ft}Hz']  = psd_at_freq(f_1k_n, psd_naive, ft)
            spectral[f'psd_proper_{ft}Hz'] = psd_at_freq(f_1k_p, psd_proper, ft)
            spectral[f'psd_res1k_{ft}Hz']  = psd_at_freq(f_res_1k, psd_res_1k, ft)
        spectral[f'psd_y20k_{ft}Hz']   = psd_at_freq(f_psd_20k, psd_y_20k, ft)
        spectral[f'psd_res20k_{ft}Hz']  = psd_at_freq(f_res_20k, psd_res_20k, ft)
        if has_delta_a:
            spectral[f'psd_da20k_{ft}Hz'] = psd_at_freq(f_psd_20k, psd_da_20k, ft)

    # Band-integrated power
    bands = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500)]
    for f_lo, f_hi in bands:
        tag = f'{f_lo}_{f_hi}'
        spectral[f'bp_y20k_{tag}']    = band_power(f_psd_20k, psd_y_20k, f_lo, f_hi)
        spectral[f'bp_res20k_{tag}']  = band_power(f_res_20k, psd_res_20k, f_lo, f_hi)
        if f_hi <= FS_NEW / 2:
            spectral[f'bp_naive_{tag}']  = band_power(f_1k_n, psd_naive, f_lo, f_hi)
            spectral[f'bp_proper_{tag}'] = band_power(f_1k_p, psd_proper, f_lo, f_hi)
            spectral[f'bp_res1k_{tag}']  = band_power(f_res_1k, psd_res_1k, f_lo, f_hi)
        if has_delta_a:
            spectral[f'bp_da20k_{tag}']  = band_power(f_psd_20k, psd_da_20k, f_lo, f_hi)

    # Above-Nyquist band power (this is what aliases in)
    above_bands = [(500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
    for f_lo, f_hi in above_bands:
        tag = f'{f_lo}_{f_hi}'
        spectral[f'bp_y20k_{tag}']   = band_power(f_psd_20k, psd_y_20k, f_lo, f_hi)
        spectral[f'bp_res20k_{tag}'] = band_power(f_res_20k, psd_res_20k, f_lo, f_hi)

    # Peak detection in residual
    # Full-band peak
    pk_f_20k, pk_v_20k = peak_in_band(f_res_20k, psd_res_20k, 1, FS_ORIG / 2)
    pk_f_1k, pk_v_1k   = peak_in_band(f_res_1k, psd_res_1k, 1, FS_NEW / 2)
    spectral['res20k_peak_freq'] = pk_f_20k
    spectral['res20k_peak_psd']  = pk_v_20k
    spectral['res1k_peak_freq']  = pk_f_1k
    spectral['res1k_peak_psd']   = pk_v_1k

    # Peak near MSD frequency (130-170 Hz band)
    pk_f_msd_20k, pk_v_msd_20k = peak_in_band(f_res_20k, psd_res_20k, 130, 170)
    pk_f_msd_1k, pk_v_msd_1k   = peak_in_band(f_res_1k, psd_res_1k, 130, 170)
    spectral['res20k_msd_peak_freq'] = pk_f_msd_20k
    spectral['res20k_msd_peak_psd']  = pk_v_msd_20k
    spectral['res1k_msd_peak_freq']  = pk_f_msd_1k
    spectral['res1k_msd_peak_psd']   = pk_v_msd_1k

    # ------------------------------------------------------------------
    # Per-trajectory 6-panel figure (3 rows x 2 cols)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    # Row 1, left: delta_a time trace
    ax = axes[0, 0]
    ax.plot(t_20k, delta_a_20k * 1e6, 'C0', lw=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('delta_a [um]')
    ax.set_title('Hidden MSD displacement (20 kHz)')
    ax.grid(True, alpha=0.3)
    if has_delta_a:
        ax.text(0.02, 0.95, f'max={da_max:.2e} m\nRMS={da_rms:.2e} m',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'delta_a not in file', transform=ax.transAxes,
                ha='center', fontsize=12, color='red')

    # Row 1, right: PSD at 20 kHz (Y output + delta_a)
    ax = axes[0, 1]
    ax.semilogy(f_psd_20k, psd_y_20k, 'C0', lw=0.8, label='Y output')
    if has_delta_a:
        ax2 = ax.twinx()
        ax2.semilogy(f_psd_20k, psd_da_20k, 'C1', lw=0.8, alpha=0.7, label='delta_a')
        ax2.set_ylabel('PSD delta_a [m^2/Hz]', color='C1', fontsize=8)
        ax2.tick_params(axis='y', labelcolor='C1')
    ax.axvline(FA_MSD, color='red', linestyle='--', alpha=0.7, label=f'fa={FA_MSD} Hz')
    ax.axvline(FS_NEW / 2, color='gray', linestyle=':', alpha=0.5,
               label=f'Nyquist @ {FS_NEW} Hz')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD Y [m^2/Hz]')
    ax.set_title('PSD at 20 kHz (full resolution)')
    ax.set_xlim([0, 1000])
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2, left: PSD at 1 kHz, naive vs proper decimation
    ax = axes[1, 0]
    ax.semilogy(f_1k_n, psd_naive,  'C0', lw=0.8, label='Naive (y[::20])')
    ax.semilogy(f_1k_p, psd_proper, 'C2', lw=0.8, label='Proper (decimate)')
    ax.axvline(FA_MSD, color='red', linestyle='--', alpha=0.7, label=f'fa={FA_MSD} Hz')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD Y [m^2/Hz]')
    ax.set_title('PSD at 1 kHz: naive vs proper decimation')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 2, right: PSD of residual at 20 kHz vs 1 kHz
    ax = axes[1, 1]
    ax.semilogy(f_res_20k, psd_res_20k, 'C0', lw=0.8, label='Residual @ 20 kHz')
    ax.semilogy(f_res_1k,  psd_res_1k,  'C3', lw=0.8, label='Residual @ 1 kHz')
    ax.axvline(FA_MSD, color='red', linestyle='--', alpha=0.7, label=f'fa={FA_MSD} Hz')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD residual [m^2/Hz]')
    ax.set_title('PSD of Y residual (measured - physics)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 3, left: time-domain Y residual at 20 kHz
    ax = axes[2, 0]
    ax.plot(t_20k[:n_cmp_20k], resid_y_20k * 1e6, 'C0', lw=0.2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Y residual [um]')
    ax.set_title('Y residual: measured - physics @ 20 kHz')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f'RMS={resid_rms_20k:.2e} m\nmax={resid_max_20k:.2e} m',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 3, right: time-domain Y residual at 1 kHz
    ax = axes[2, 1]
    ax.plot(t_1k[:n_cmp_1k], resid_y_1k * 1e6, 'C3', lw=0.3)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Y residual [um]')
    ax.set_title('Y residual: measured - physics @ 1 kHz')
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.95, f'RMS={resid_rms_1k:.2e} m\nmax={resid_max_1k:.2e} m',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(f'{label}: MSD visibility check (fa={FA_MSD} Hz)', fontsize=11)
    fig.tight_layout()
    path = os.path.join(save_dir, f'msd_visibility_{label}_{run_id}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        'label': label,
        'fname': fname,
        'has_delta_a': has_delta_a,
        'da_max': da_max,
        'da_rms': da_rms,
        'resid_rms_y_20k': resid_rms_20k,
        'resid_max_y_20k': resid_max_20k,
        'resid_rms_ch_20k': resid_rms_ch_20k.copy(),
        'resid_rms_y_1k': resid_rms_1k,
        'resid_max_y_1k': resid_max_1k,
        'resid_rms_ch_1k': resid_rms_ch_1k.copy(),
        'ratio': ratio,
        'spectral': spectral,
    }


# =========================================================================
# Main
# =========================================================================

if __name__ == '__main__':
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_id = (os.environ.get('SLURM_JOB_ID')
              or datetime.now().strftime('%Y%m%d_%H%M%S'))

    traj_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'data', 'gantry', 'matlab', DATA_SUBDIR)
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'simulations', 'gantry_subnet', 'msd_verification')
    os.makedirs(save_dir, exist_ok=True)

    print(f"MSD visibility verification (fa={FA_MSD} Hz, ma={MA_FRAC*100:.0f}% of mh)")
    print(f"Mode: {MODE}  (data from {DATA_SUBDIR}/)")
    print(f"Decimation: {FS_ORIG} Hz -> {FS_NEW} Hz (factor {D}, no anti-alias filter)")
    print(f"Run ID: {run_id}")

    # Compute normalization once in the main process
    print("Computing normalization stats...", flush=True)
    norm = compute_normalization(traj_dir)
    print("Done.\n")

    # Parallel processing: one process per trajectory
    n_workers = min(len(ALL_FILES), os.cpu_count() or 1)
    print(f"Processing {len(ALL_FILES)} trajectories on {n_workers} workers...\n")

    worker = partial(process_trajectory,
                     norm=norm, traj_dir=traj_dir,
                     save_dir=save_dir, run_id=run_id)

    with multiprocessing.Pool(n_workers) as pool:
        results = pool.map(worker, ALL_FILES)

    # Print per-trajectory table
    print(f"\n{'Trajectory':<42s}  {'delta_a max':>12s}  {'delta_a RMS':>12s}  "
          f"{'Y res 20k':>10s}  {'Y res 1k':>10s}  {'ratio':>6s}")
    for r in results:
        print(f"  {r['label']:<40s}  {r['da_max']:12.2e}  {r['da_rms']:12.2e}  "
              f"{r['resid_rms_y_20k']:10.2e}  {r['resid_rms_y_1k']:10.2e}  "
              f"{r['ratio']:6.2f}")

    # =====================================================================
    # Summary bar chart
    # =====================================================================
    labels = [r['label'] for r in results]
    da_rms_all     = np.array([r['da_rms'] for r in results])
    resid_rms_20k  = np.array([r['resid_rms_y_20k'] for r in results])
    resid_rms_1k   = np.array([r['resid_rms_y_1k'] for r in results])
    resid_ch_20k   = np.array([r['resid_rms_ch_20k'] for r in results])
    resid_ch_1k    = np.array([r['resid_rms_ch_1k'] for r in results])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_pos = np.arange(len(labels))
    bar_w = 0.35

    # delta_a RMS
    ax = axes[0]
    ax.bar(x_pos, da_rms_all * 1e6, color='C0')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('delta_a RMS [um]')
    ax.set_title('MSD displacement RMS')
    ax.grid(True, alpha=0.3, axis='y')

    # Y-channel residual RMS: 20 kHz vs 1 kHz
    ax = axes[1]
    ax.bar(x_pos - bar_w/2, resid_rms_20k * 1e6, bar_w, label='@ 20 kHz', color='C0')
    ax.bar(x_pos + bar_w/2, resid_rms_1k * 1e6,  bar_w, label='@ 1 kHz',  color='C3')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Y residual RMS [um]')
    ax.set_title('Y residual RMS: 20 kHz vs 1 kHz')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # Per-channel residual RMS at 20 kHz
    ax = axes[2]
    bar_w_ch = 0.13
    ch_labels = ['X1', 'X2', 'Y']
    for ch in range(3):
        ax.bar(x_pos + ch * bar_w_ch - bar_w_ch * 2.5,
               resid_ch_20k[:, ch] * 1e6, bar_w_ch,
               label=f'{ch_labels[ch]} 20k', color=f'C{ch}')
        ax.bar(x_pos + ch * bar_w_ch - bar_w_ch * 2.5 + 3 * bar_w_ch,
               resid_ch_1k[:, ch] * 1e6, bar_w_ch,
               label=f'{ch_labels[ch]} 1k', color=f'C{ch}', alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Residual RMS [um]')
    ax.set_title('Per-channel residual: 20 kHz (solid) vs 1 kHz (faded)')
    ax.legend(fontsize=5, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'MSD visibility summary (fa={FA_MSD} Hz, decimation {FS_ORIG}->{FS_NEW} Hz)',
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(save_dir, f'msd_summary_{run_id}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {os.path.basename(path)}")

    # =====================================================================
    # Save .npz
    # =====================================================================
    save_dict = {
        'run_id': run_id,
        'FA_MSD': FA_MSD,
        'FS_ORIG': FS_ORIG,
        'FS_NEW': FS_NEW,
        'D': D,
        'n_trajectories': len(results),
        'labels': np.array([r['label'] for r in results]),
        'da_max': np.array([r['da_max'] for r in results]),
        'da_rms': da_rms_all,
        'resid_rms_y_20k': resid_rms_20k,
        'resid_max_y_20k': np.array([r['resid_max_y_20k'] for r in results]),
        'resid_rms_ch_20k': resid_ch_20k,
        'resid_rms_y_1k': resid_rms_1k,
        'resid_max_y_1k': np.array([r['resid_max_y_1k'] for r in results]),
        'resid_rms_ch_1k': resid_ch_1k,
    }
    path = os.path.join(save_dir, f'msd_results_{run_id}.npz')
    np.savez(path, **save_dict)
    print(f"  Saved: {os.path.basename(path)}")

    # =====================================================================
    # TABLE 1: Summary
    # =====================================================================
    print(f"\n{'='*90}")
    print("TABLE 1: SUMMARY")
    print(f"{'='*90}")
    print(f"  MSD natural frequency:       {FA_MSD} Hz")
    print(f"  Nyquist after decimation:     {FS_NEW/2:.0f} Hz")
    print(f"  Decimation factor:            {D} (naive stride, no anti-alias)")
    print(f"")
    print(f"  delta_a RMS (mean):           {np.mean(da_rms_all):.2e} m = {np.mean(da_rms_all)*1e6:.2f} um")
    print(f"  delta_a max (worst):          {np.max([r['da_max'] for r in results]):.2e} m")
    print(f"")
    print(f"  Y residual RMS @ 20 kHz:      {np.mean(resid_rms_20k):.2e} m = {np.mean(resid_rms_20k)*1e6:.2f} um")
    print(f"  Y residual RMS @ 1 kHz:       {np.mean(resid_rms_1k):.2e} m = {np.mean(resid_rms_1k)*1e6:.2f} um")
    print(f"  Ratio (1k / 20k):             {np.mean(resid_rms_1k) / np.mean(resid_rms_20k):.2f}")

    print(f"\n  Interpretation:")
    if np.mean(da_rms_all) < 1e-7:
        print(f"  - MSD barely excited (delta_a < 0.1 um RMS).")
        print(f"    The hidden mass has negligible effect on measured outputs.")
    else:
        r_ratio = np.mean(resid_rms_1k) / np.mean(resid_rms_20k)
        if np.mean(resid_rms_20k) < 1e-6:
            print(f"  - MSD excited but Y residual < 1 um RMS even at 20 kHz.")
            print(f"    The MSD effect on stage positions is inherently tiny.")
        elif r_ratio < 0.5:
            print(f"  - 20 kHz residual is {np.mean(resid_rms_20k)*1e6:.1f} um but")
            print(f"    1 kHz residual drops to {np.mean(resid_rms_1k)*1e6:.1f} um.")
            print(f"    Decimation is destroying the MSD signal.")
        elif r_ratio > 1.5:
            print(f"  - 1 kHz residual ({np.mean(resid_rms_1k)*1e6:.1f} um) is LARGER than")
            print(f"    20 kHz residual ({np.mean(resid_rms_20k)*1e6:.1f} um).")
            print(f"    Aliasing from naive decimation is adding spurious energy.")
        else:
            print(f"  - Residuals are comparable at both rates (ratio={r_ratio:.2f}).")
            print(f"    MSD signal survives decimation.")

    # =====================================================================
    # TABLE 2: Per-channel residual RMS
    # =====================================================================
    print(f"\n{'='*90}")
    print("TABLE 2: PER-CHANNEL RESIDUAL RMS [um]")
    print(f"{'='*90}")
    ch_names = ['X1', 'X2', 'Y']
    header = f"  {'Trajectory':<30s}"
    for ch in ch_names:
        header += f"  {ch+' 20k':>8s}  {ch+' 1k':>8s}  {'ratio':>6s}"
    print(header)
    for r in results:
        row = f"  {r['label']:<30s}"
        for ch in range(3):
            v20 = r['resid_rms_ch_20k'][ch]
            v1  = r['resid_rms_ch_1k'][ch]
            rat = v1 / v20 if v20 > 0 else float('inf')
            row += f"  {v20*1e6:8.2f}  {v1*1e6:8.2f}  {rat:6.2f}"
        print(row)

    # =====================================================================
    # TABLE 3: Residual peak frequencies
    # =====================================================================
    print(f"\n{'='*90}")
    print("TABLE 3: RESIDUAL PEAK DETECTION")
    print(f"{'='*90}")
    print(f"  Where is most residual energy? (full-band peak and peak near {FA_MSD} Hz)")
    print(f"  {'Trajectory':<30s}  {'pk_f 20k':>8s}  {'pk_dB 20k':>9s}  "
          f"{'pk_f 1k':>8s}  {'pk_dB 1k':>9s}  "
          f"{'msd_f 20k':>9s}  {'msd_dB 20k':>10s}  "
          f"{'msd_f 1k':>9s}  {'msd_dB 1k':>10s}")
    for r in results:
        sp = r['spectral']
        pk20_f = sp['res20k_peak_freq']
        pk20_v = 10 * np.log10(sp['res20k_peak_psd'] + 1e-30)
        pk1_f  = sp['res1k_peak_freq']
        pk1_v  = 10 * np.log10(sp['res1k_peak_psd'] + 1e-30)
        msd20_f = sp['res20k_msd_peak_freq']
        msd20_v = 10 * np.log10(sp['res20k_msd_peak_psd'] + 1e-30)
        msd1_f  = sp['res1k_msd_peak_freq']
        msd1_v  = 10 * np.log10(sp['res1k_msd_peak_psd'] + 1e-30)
        print(f"  {r['label']:<30s}  {pk20_f:7.1f}Hz  {pk20_v:9.1f}  "
              f"{pk1_f:7.1f}Hz  {pk1_v:9.1f}  "
              f"{msd20_f:8.1f}Hz  {msd20_v:10.1f}  "
              f"{msd1_f:8.1f}Hz  {msd1_v:10.1f}")

    # =====================================================================
    # TABLE 4: Band-integrated power — residual (20 kHz vs 1 kHz)
    # =====================================================================
    bands = [(0, 50), (50, 100), (100, 200), (200, 300), (300, 500)]
    print(f"\n{'='*90}")
    print("TABLE 4: BAND-INTEGRATED RESIDUAL POWER [dB re 1 m^2]")
    print(f"{'='*90}")
    print(f"  Compares residual energy per frequency band at 20 kHz vs 1 kHz.")
    print(f"  Ratio > 0 dB means 1 kHz has MORE residual energy (aliasing adding energy).")
    header = f"  {'Trajectory':<20s}"
    for f_lo, f_hi in bands:
        header += f"  {f_lo}-{f_hi}Hz 20k  {f_lo}-{f_hi}Hz 1k  {'diff':>5s}"
    print(header)
    for r in results:
        sp = r['spectral']
        row = f"  {r['label']:<20s}"
        for f_lo, f_hi in bands:
            tag = f'{f_lo}_{f_hi}'
            bp20 = sp.get(f'bp_res20k_{tag}', 0)
            bp1  = sp.get(f'bp_res1k_{tag}', 0)
            db20 = 10 * np.log10(bp20 + 1e-30)
            db1  = 10 * np.log10(bp1 + 1e-30)
            row += f"  {db20:11.1f}  {db1:10.1f}  {db1-db20:+5.1f}"
        print(row)

    # =====================================================================
    # TABLE 5: Naive vs proper decimation — band-integrated power
    # =====================================================================
    print(f"\n{'='*90}")
    print("TABLE 5: NAIVE vs PROPER DECIMATION — BAND POWER of Y-channel [dB re 1 m^2]")
    print(f"{'='*90}")
    print(f"  Positive diff = naive has MORE energy = aliased content from above Nyquist.")
    header = f"  {'Trajectory':<20s}"
    for f_lo, f_hi in bands:
        header += f"  {f_lo}-{f_hi} N  {f_lo}-{f_hi} P  {'diff':>5s}"
    print(header)
    for r in results:
        sp = r['spectral']
        row = f"  {r['label']:<20s}"
        for f_lo, f_hi in bands:
            tag = f'{f_lo}_{f_hi}'
            bn = sp.get(f'bp_naive_{tag}', 0)
            bp = sp.get(f'bp_proper_{tag}', 0)
            db_n = 10 * np.log10(bn + 1e-30)
            db_p = 10 * np.log10(bp + 1e-30)
            row += f"  {db_n:10.1f}  {db_p:10.1f}  {db_n-db_p:+5.1f}"
        print(row)

    # =====================================================================
    # TABLE 6: Above-Nyquist band power in 20 kHz data (aliasing source)
    # =====================================================================
    above_bands = [(500, 1000), (1000, 2000), (2000, 5000), (5000, 10000)]
    print(f"\n{'='*90}")
    print(f"TABLE 6: ABOVE-NYQUIST BAND POWER in 20 kHz data [dB re 1 m^2]")
    print(f"{'='*90}")
    print(f"  Energy above {FS_NEW/2:.0f} Hz that folds into 0-{FS_NEW/2:.0f} Hz during naive decimation.")
    print(f"  Compare to in-band (0-500 Hz) to judge aliasing severity.")
    header = f"  {'Trajectory':<20s}  {'0-500 Y':>9s}  {'0-500 res':>9s}"
    for f_lo, f_hi in above_bands:
        header += f"  {f_lo/1000:.0f}-{f_hi/1000:.0f}k Y  {f_lo/1000:.0f}-{f_hi/1000:.0f}k res"
    print(header)
    for r in results:
        sp = r['spectral']
        # In-band reference
        inband_y = sum(sp.get(f'bp_y20k_{fl}_{fh}', 0)
                       for fl, fh in bands)
        inband_r = sum(sp.get(f'bp_res20k_{fl}_{fh}', 0)
                       for fl, fh in bands)
        row = f"  {r['label']:<20s}"
        row += f"  {10*np.log10(inband_y+1e-30):9.1f}  {10*np.log10(inband_r+1e-30):9.1f}"
        for f_lo, f_hi in above_bands:
            tag = f'{f_lo}_{f_hi}'
            by = sp.get(f'bp_y20k_{tag}', 0)
            br = sp.get(f'bp_res20k_{tag}', 0)
            row += f"  {10*np.log10(by+1e-30):8.1f}  {10*np.log10(br+1e-30):8.1f}"
        print(row)

    # =====================================================================
    # TABLE 7: delta_a spectral content
    # =====================================================================
    has_any_da = any(r['has_delta_a'] for r in results)
    if has_any_da:
        print(f"\n{'='*90}")
        print("TABLE 7: DELTA_A (MSD DISPLACEMENT) SPECTRAL CONTENT [dB re 1 m^2]")
        print(f"{'='*90}")
        print(f"  Band power of delta_a at 20 kHz — shows where MSD energy is concentrated.")
        header = f"  {'Trajectory':<20s}  {'da RMS um':>9s}"
        for f_lo, f_hi in bands:
            header += f"  {f_lo}-{f_hi}Hz"
        print(header)
        for r in results:
            if not r['has_delta_a']:
                continue
            sp = r['spectral']
            row = f"  {r['label']:<20s}  {r['da_rms']*1e6:9.2f}"
            for f_lo, f_hi in bands:
                tag = f'{f_lo}_{f_hi}'
                bp = sp.get(f'bp_da20k_{tag}', 0)
                row += f"  {10*np.log10(bp+1e-30):9.1f}"
            print(row)

    # =====================================================================
    # TABLE 8: Point-sample PSD at MSD frequency — all signals
    # =====================================================================
    print(f"\n{'='*90}")
    print(f"TABLE 8: PSD AT {FA_MSD} Hz — ALL SIGNALS [dB re 1 m^2/Hz]")
    print(f"{'='*90}")
    print(f"  {'Trajectory':<20s}  {'Y@20k':>8s}  {'da@20k':>8s}  {'res@20k':>8s}  "
          f"{'naive@1k':>8s}  {'proper@1k':>9s}  {'N-P diff':>8s}  {'res@1k':>8s}  "
          f"{'res 1k-20k':>10s}")
    for r in results:
        sp = r['spectral']
        y20   = 10 * np.log10(sp[f'psd_y20k_{FA_MSD}Hz'] + 1e-30)
        r20   = 10 * np.log10(sp[f'psd_res20k_{FA_MSD}Hz'] + 1e-30)
        naive = 10 * np.log10(sp[f'psd_naive_{FA_MSD}Hz'] + 1e-30)
        prop  = 10 * np.log10(sp[f'psd_proper_{FA_MSD}Hz'] + 1e-30)
        r1    = 10 * np.log10(sp[f'psd_res1k_{FA_MSD}Hz'] + 1e-30)
        da_str = '     N/A'
        if r['has_delta_a'] and f'psd_da20k_{FA_MSD}Hz' in sp:
            da_str = f"{10*np.log10(sp[f'psd_da20k_{FA_MSD}Hz']+1e-30):8.1f}"
        print(f"  {r['label']:<20s}  {y20:8.1f}  {da_str}  {r20:8.1f}  "
              f"{naive:8.1f}  {prop:9.1f}  {naive-prop:+8.1f}  {r1:8.1f}  "
              f"{r1-r20:+10.1f}")
