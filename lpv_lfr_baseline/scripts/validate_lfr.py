"""
validate_lfr.py
---------------
Visual validation and printed diagnostics for the dual-gantry LPV-LFR baseline.

Five plots with accompanying printed summaries:

  [1] Trajectory comparison
      Python RK4 vs MATLAB quasi-LPV ODE (lpv_sim_varying_y.mat).
      Y moves 0.3 -> -0.3 m. Expected error ~2e-14 m (same CT ODE, RK4 vs ode45).

  [2] Bode at fixed Y = 0.3 m
      Python CT FRF vs MATLAB ZOH DT (gantry_G_matrices.mat).
      Full 3x3 magnitude grid. CT and ZOH overlay closely below ~1 kHz.

  [3] Y-varying Bode family
      Python CT FRF at Y in {-0.30, -0.15, 0.00, +0.15, +0.30} m.
      MATLAB ZOH DT at nearest available Y (lpv_matrices.mat) overlaid as dashed.
      Shows the LPV scheduling effect: how dynamics shift with payload position.

  [4] Natural frequencies vs Y
      The 3 natural frequencies of A_c(Y) swept over Y in [-0.4, +0.4] m.
      Shows how resonant frequencies shift with payload position — motivates LPV.

  [5] LPV vs frozen LTI trajectory comparison
      Python LPV and Python frozen LTI (M frozen at Y=0.3), both driven by u_q1.
      Reference: q1 (MATLAB quasi-LPV). Metrics: BFR, RMSE, Max|error| per channel.
      Demonstrates LPV scheduling benefit over frozen baseline.

Usage:
    conda run -n GraduationProject python -m lpv_lfr_baseline.validate_lfr
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

from lpv_lfr_baseline.core.physics import (
    M0, M1, M2, K, C, P, fs, ts, build_M, build_poly_constants,
    mh as _mh, m1 as _m1, m2 as _m2, mb as _mb, Jb as _Jb, Jh as _Jh,
    Lb as _Lb, d as _d,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_forward import lfr_forward
from lpv_lfr_baseline.core.lfr_simulate import simulate, simulate_frozen, SimResult

# Precompute G and poly constants from fixed physics params once at module load
_G_TRUE    = build_G_matrix(M0, M1, M2, K, C)
_alpha, _beta, _gamma, _N0, _N1, _N2 = build_poly_constants(
    _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_MAT_BASE    = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab-output')
_DTYPE       = torch.float64
_FS          = fs.item()
_TS          = ts.item()
_FREQS_HZ    = np.logspace(np.log10(1.0), np.log10(500.0), 1000)   # 1-500 Hz, log
_Y_FIXED     = 0.300                              # m, for Plot 2
_Y_FAMILY    = [-0.30, -0.15, 0.00, 0.15, 0.30]  # m, for Plot 3
_Y_FREEZE    = 0.300                              # m, frozen LTI operating point (Plot 5)
_Y_SWEEP     = np.linspace(-0.4, 0.4, 200)        # m, for Plot 4 (nat freqs vs Y)
_YLIM_DB     = (-80, 20)                          # dB axis limits for Bode plots
_IN_LABELS   = ['$F_{X1}$', '$F_{X2}$', '$F_Y$']
_OUT_LABELS  = ['$X_1$',    '$X_2$',    '$Y$']
_CH_NAMES    = ['X1', 'X2', 'Y ']
_SEP         = '=' * 72


# ---------------------------------------------------------------------------
# FRF helpers
# ---------------------------------------------------------------------------

def _ct_frf_stage(Y_val: float) -> np.ndarray:
    """
    Continuous-time FRF in stage coordinates at scalar Y [m].

    H_stage(jw) = P.T @ H_logical(jw) @ P
    H_logical   = C_c (jwI - A_c)^-1 B_c   (logical coords, D=0)

    Returns H : (nf, 3, 3) complex128, output x input.
    """
    Y_t   = torch.tensor(Y_val, dtype=_DTYPE)
    M_Y   = build_M(Y_t)
    eye3  = torch.eye(3, dtype=_DTYPE)

    MYinvK = torch.linalg.solve(M_Y, K).numpy()
    MYinvC = torch.linalg.solve(M_Y, C).numpy()
    MYinv  = torch.linalg.solve(M_Y, eye3).numpy()

    A_c = np.block([[np.zeros((3, 3)), np.eye(3)   ],
                    [-MYinvK,          -MYinvC      ]])  # (6, 6)
    B_c = np.vstack([np.zeros((3, 3)), MYinv        ])   # (6, 3) logical coords
    C_c = np.hstack([np.eye(3),        np.zeros((3, 3))])  # (3, 6) logical coords

    P_np = P.numpy()
    nf   = len(_FREQS_HZ)
    jw   = 1j * 2 * np.pi * _FREQS_HZ                      # (nf,)

    # (nf, 6, 6): jw*I - A_c
    sys_mat   = jw[:, None, None] * np.eye(6) - A_c         # broadcast (nf,6,6)
    resolvent = np.linalg.solve(sys_mat,
                                np.broadcast_to(B_c, (nf, 6, 3)).copy())  # (nf,6,3)
    H_logical = C_c @ resolvent                              # (nf, 3, 3)

    # Stage coordinate transform: H_stage = P.T @ H_logical @ P
    H_stage = P_np.T @ H_logical @ P_np                     # numpy broadcasts (nf,3,3)
    return H_stage


def _dt_frf_stage(A_d: np.ndarray, B_d: np.ndarray,
                  C_d: np.ndarray, D_d: np.ndarray) -> np.ndarray:
    """
    Discrete-time FRF: H(e^{jwTs}) = C (e^{jwTs}I - A)^-1 B + D.
    Matrices assumed already in stage coordinates.
    Returns H : (nf, 3, 3) complex128.
    """
    nf  = len(_FREQS_HZ)
    z   = np.exp(1j * 2 * np.pi * _FREQS_HZ * _TS)          # (nf,)
    sys_mat   = z[:, None, None] * np.eye(6) - A_d           # (nf, 6, 6)
    resolvent = np.linalg.solve(sys_mat,
                                np.broadcast_to(B_d, (nf, 6, 3)).copy())  # (nf,6,3)
    H = C_d @ resolvent + D_d                                # (nf, 3, 3)
    return H


def _mag_db(H: np.ndarray) -> np.ndarray:
    """Magnitude in dB; clip -inf to _YLIM_DB[0]."""
    return np.clip(20 * np.log10(np.abs(H) + 1e-300), _YLIM_DB[0], _YLIM_DB[1])


def _nat_freqs_hz(Y_val: float) -> np.ndarray:
    """Natural frequencies [Hz] from poles of A_c(Y). Returns sorted unique values > 1 Hz."""
    Y_t    = torch.tensor(Y_val, dtype=_DTYPE)
    M_Y    = build_M(Y_t)
    eye3   = torch.eye(3, dtype=_DTYPE)
    MYinvK = torch.linalg.solve(M_Y, K).numpy()
    MYinvC = torch.linalg.solve(M_Y, C).numpy()
    A_c    = np.block([[np.zeros((3, 3)), np.eye(3)],
                       [-MYinvK,          -MYinvC  ]])
    eigs   = np.linalg.eigvals(A_c)
    freqs  = np.abs(eigs.imag) / (2 * np.pi)
    return np.sort(np.unique(np.round(freqs[freqs > 1.0], 2)))


def _nearest_mat_idx(Y_values: np.ndarray, Y_target: float) -> int:
    """Index into Y_values closest to Y_target."""
    return int(np.argmin(np.abs(Y_values.squeeze() - Y_target)))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _phase_unwrapped(H: np.ndarray, row: int, col: int) -> np.ndarray:
    """Unwrapped phase in degrees for H[:, row, col]."""
    return np.degrees(np.unwrap(np.angle(H[:, row, col])))


def _bode_axes(axes, row: int, col: int,
               freqs: np.ndarray, mag_db: np.ndarray,
               label: str, color: str, linestyle: str = '-', lw: float = 1.5):
    ax = axes[row, col]
    ax.semilogx(freqs, mag_db[:, row, col], color=color,
                linestyle=linestyle, linewidth=lw, label=label)
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_ylim(_YLIM_DB)
    ax.grid(True, which='both', alpha=0.3)
    if col == 0:
        ax.set_ylabel('Magnitude [dB]')
    ax.set_title(f'{_OUT_LABELS[row]} \u2190 {_IN_LABELS[col]}', fontsize=9)


def _phase_axes(axes, col: int,
                freqs: np.ndarray, H: np.ndarray,
                label: str, color: str, linestyle: str = '-', lw: float = 1.5):
    """Plot unwrapped phase for diagonal channel [col, col] into phase row (row=3)."""
    ax = axes[3, col]
    phase = _phase_unwrapped(H, col, col)
    ax.semilogx(freqs, phase, color=color, linestyle=linestyle, linewidth=lw, label=label)
    ax.set_xlim([freqs[0], freqs[-1]])
    ax.set_ylim([-400, 20])
    ax.set_yticks([-360, -270, -180, -90, 0])
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlabel('Frequency [Hz]')
    if col == 0:
        ax.set_ylabel('Phase [deg]')
    ax.set_title(f'{_OUT_LABELS[col]} \u2190 {_IN_LABELS[col]}  (phase)', fontsize=9)


# ---------------------------------------------------------------------------
# Section 1: Trajectory comparison
# ---------------------------------------------------------------------------

def _section_trajectory():
    path = os.path.join(_MAT_BASE, 'lpv_sim_varying_y.mat')
    if not os.path.exists(path):
        print(f'  SKIPPED - file not found: {path}')
        return None

    mat    = loadmat(path)
    q1_ref = torch.tensor(mat['q1'],   dtype=_DTYPE)   # (N, 3) stage coords
    u_seq  = torch.tensor(mat['u_q1'], dtype=_DTYPE)   # (N, 3) stage forces
    t_sim  = mat['t_sim'].squeeze()                     # (N,)
    N      = q1_ref.shape[0]

    # Initial state: Y=0.3 m, all else zero (logical coords)
    x0 = torch.zeros(1, 6, dtype=_DTYPE)
    x0[0, 2] = 0.3

    with torch.no_grad():
        result = simulate(
            x0, u_seq.unsqueeze(0),
            _G_TRUE, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2, P, ts,
        )

    Y_py  = result.Y[0]              # (N, 3) stage coords: Python
    err   = (Y_py - q1_ref).abs()   # (N, 3)

    # --- Print ---
    print(_SEP)
    print('[1/5]  TRAJECTORY  -  Python RK4 vs MATLAB quasi-LPV ODE')
    print(_SEP)
    print(f'  Source    : lpv_sim_varying_y.mat')
    print(f'  Steps     : {N}       Duration : {N / _FS:.3f} s')
    y_py_vals = result.X[0, :, 2]
    print(f'  Y range   : {y_py_vals.min().item():.3f} to {y_py_vals.max().item():.3f} m')
    print()
    ch_names = ['X1', 'X2', 'Y ']
    print(f"  {'Channel':12s}  {'Max |error| [m]':18s}  {'Mean |error| [m]':18s}")
    print(f"  {'-'*12}  {'-'*18}  {'-'*18}")
    for i, name in enumerate(ch_names):
        print(f'  {name:12s}  {err[:, i].max().item():18.3e}  {err[:, i].mean().item():18.3e}')
    overall = err.max().item()
    status  = 'PASS' if overall < 1e-10 else 'FAIL'
    print(f"  {'-'*12}  {'-'*18}")
    print(f'  {"Overall":12s}  {overall:.3e}  (threshold 1e-10 m)   {status}')
    print()

    # --- Plot ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Trajectory: Python RK4 vs MATLAB quasi-LPV ODE (varying Y)', fontsize=12)

    for i, (name, ax) in enumerate(zip(['X1 [m]', 'X2 [m]', 'Y [m]'], axes[:3])):
        ax.plot(t_sim, q1_ref[:, i].numpy(), color='tab:orange',
                linestyle='--', linewidth=1.5, label='MATLAB (ode45)')
        ax.plot(t_sim, Y_py[:, i].numpy(), color='tab:blue',
                linewidth=1.0, label='Python (RK4)', alpha=0.85)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    axes[3].semilogy(t_sim, err[:, 0].numpy(), label='X1', color='tab:blue')
    axes[3].semilogy(t_sim, err[:, 1].numpy(), label='X2', color='tab:orange')
    axes[3].semilogy(t_sim, err[:, 2].numpy(), label='Y',  color='tab:green')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_ylabel('|error| [m]')
    axes[3].set_title('Absolute position error per channel')
    axes[3].grid(True, which='both', alpha=0.3)
    axes[3].legend(fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section 2: Bode at fixed Y = 0.3 m  (Python CT vs MATLAB ZOH DT)
# ---------------------------------------------------------------------------

def _section_bode_fixed_y():
    path = os.path.join(_MAT_BASE, 'gantry_G_matrices.mat')
    if not os.path.exists(path):
        print(f'  SKIPPED - file not found: {path}')
        return None

    mat  = loadmat(path)
    A_d  = mat['A'].astype(np.float64)
    B_d  = mat['B'].astype(np.float64)
    C_d  = mat['C'].astype(np.float64)
    D_d  = mat['D'].astype(np.float64)

    H_ct = _ct_frf_stage(_Y_FIXED)
    H_dt = _dt_frf_stage(A_d, B_d, C_d, D_d)
    mag_ct = _mag_db(H_ct)
    mag_dt = _mag_db(H_dt)

    nat_f = _nat_freqs_hz(_Y_FIXED)

    # --- Print ---
    print(_SEP)
    print(f'[2/5]  BODE  -  Python CT vs MATLAB ZOH DT  at  Y = {_Y_FIXED:.3f} m')
    print(_SEP)
    print(f'  Source        : gantry_G_matrices.mat')
    print(f'  Freq range    : {_FREQS_HZ[0]:.1f} - {_FREQS_HZ[-1]:.0f} Hz  '
          f'({len(_FREQS_HZ)} points, log-spaced)')
    print(f'  Natural freqs : {nat_f} Hz')
    print()
    print('  Max |CT mag - ZOH DT mag| [dB] per channel (output <- input):')
    header = '  {:12s}  ' + '  '.join([f'{lb:>10s}' for lb in ['F_X1', 'F_X2', 'F_Y']])
    print(header.format(''))
    print('  ' + '-' * 50)
    out_names = ['X1', 'X2', 'Y ']
    for i, oname in enumerate(out_names):
        devs = [f'{np.max(np.abs(mag_ct[:, i, j] - mag_dt[:, i, j])):10.4f}'
                for j in range(3)]
        print(f'  {oname:12s}  ' + '  '.join(devs))
    print()

    # Also print peak frequencies from diagonal CT FRFs
    print('  Diagonal peak frequencies (CT, magnitude maxima above 2 Hz):')
    for i in range(3):
        diag_db  = mag_ct[:, i, i]
        mask     = _FREQS_HZ > 2.0
        idx_peak = np.argmax(diag_db[mask])
        f_peak   = _FREQS_HZ[mask][idx_peak]
        m_peak   = diag_db[mask][idx_peak]
        print(f'  {out_names[i]:12s} <- {["F_X1","F_X2","F_Y"][i]:6s}  '
              f'peak {m_peak:.1f} dB  at {f_peak:.2f} Hz')
    print()

    # --- Plot ---
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Bode: Python CT vs MATLAB ZOH DT  |  Y = {_Y_FIXED:.2f} m  '
                 f'(row 4 = diagonal phase)', fontsize=11)

    for row in range(3):
        for col in range(3):
            _bode_axes(axes, row, col, _FREQS_HZ, mag_dt,
                       'MATLAB ZOH DT', 'tab:orange', '--', lw=1.8)
            _bode_axes(axes, row, col, _FREQS_HZ, mag_ct,
                       'Python CT', 'tab:blue', '-', lw=1.4)
            for f in nat_f:
                axes[row, col].axvline(f, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    # Phase row — diagonal channels only
    for col in range(3):
        _phase_axes(axes, col, _FREQS_HZ, H_dt,
                    'MATLAB ZOH DT', 'tab:orange', '--', lw=1.8)
        _phase_axes(axes, col, _FREQS_HZ, H_ct,
                    'Python CT', 'tab:blue', '-', lw=1.4)
        for f in nat_f:
            axes[3, col].axvline(f, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

    axes[0, 2].legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section 3: Y-varying Bode family
# ---------------------------------------------------------------------------

def _section_bode_varying_y():
    path = os.path.join(_MAT_BASE, 'lpv_matrices.mat')
    if not os.path.exists(path):
        print(f'  SKIPPED - file not found: {path}')
        return None

    mat      = loadmat(path)
    A_all    = mat['A_all']    # (6, 6, 50)
    B_all    = mat['B_all']    # (6, 3, 50)
    C_all    = mat['C_all']    # (3, 6, 50)
    D_all    = mat['D_all'].astype(np.float64)  # (3, 3, 50)
    Y_values = mat['Y_values'].squeeze()        # (50,)

    cmap   = plt.get_cmap('coolwarm')
    colors = [cmap(i / (len(_Y_FAMILY) - 1)) for i in range(len(_Y_FAMILY))]

    # --- Print ---
    print(_SEP)
    print('[3/5]  Y-VARYING BODE  -  Python CT FRF at 5 scheduling points')
    print(_SEP)
    print(f'  Source        : lpv_matrices.mat')
    print(f'  Y values [m]  : {_Y_FAMILY}')
    print(f'  Freq range    : {_FREQS_HZ[0]:.1f} - {_FREQS_HZ[-1]:.0f} Hz')
    print()
    print('  Diagonal peak frequency [Hz] and magnitude [dB] per Y value:')
    print(f"  {'Y [m]':>8s}  {'X1<-F_X1 (Hz / dB)':>22s}  "
          f"{'X2<-F_X2 (Hz / dB)':>22s}  {'Y<-F_Y (Hz / dB)':>20s}")
    print('  ' + '-' * 82)

    # --- Plot ---
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), sharex=True)
    fig.suptitle('Y-varying Bode: Python CT FRF  |  dashed = MATLAB ZOH DT  '
                 '(row 4 = diagonal phase)', fontsize=11)

    H_ct_cache = {}
    for idx, Y_val in enumerate(sorted(_Y_FAMILY)):
        H_ct      = _ct_frf_stage(Y_val)
        H_ct_cache[Y_val] = H_ct
        mag_ct    = _mag_db(H_ct)
        col_c     = colors[idx]
        label_ct  = f'CT  Y={Y_val:+.2f}'

        # MATLAB DT at nearest Y
        near_i = _nearest_mat_idx(Y_values, Y_val)
        H_dt   = _dt_frf_stage(A_all[:, :, near_i], B_all[:, :, near_i],
                                C_all[:, :, near_i], D_all[:, :, near_i])
        mag_dt = _mag_db(H_dt)

        for row in range(3):
            for col in range(3):
                ax = axes[row, col]
                ax.semilogx(_FREQS_HZ, mag_dt[:, row, col],
                            color=col_c, linestyle='--', linewidth=1.0, alpha=0.6)
                ax.semilogx(_FREQS_HZ, mag_ct[:, row, col],
                            color=col_c, linestyle='-', linewidth=1.5,
                            label=label_ct if (row == 0 and col == 2) else '_nolegend_')
                ax.set_xlim([_FREQS_HZ[0], _FREQS_HZ[-1]])
                ax.set_ylim(_YLIM_DB)
                ax.grid(True, which='both', alpha=0.3)
                if col == 0:
                    ax.set_ylabel('Magnitude [dB]')
                ax.set_title(f'{_OUT_LABELS[row]} \u2190 {_IN_LABELS[col]}', fontsize=9)

        # Phase row — diagonal channels
        for col in range(3):
            ax = axes[3, col]
            phase_ct = _phase_unwrapped(H_ct, col, col)
            phase_dt = _phase_unwrapped(H_dt, col, col)
            ax.semilogx(_FREQS_HZ, phase_dt, color=col_c, linestyle='--',
                        linewidth=1.0, alpha=0.6)
            ax.semilogx(_FREQS_HZ, phase_ct, color=col_c, linestyle='-', linewidth=1.5)
            ax.set_xlim([_FREQS_HZ[0], _FREQS_HZ[-1]])
            ax.set_ylim([-400, 20])
            ax.set_yticks([-360, -270, -180, -90, 0])
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlabel('Frequency [Hz]')
            if col == 0:
                ax.set_ylabel('Phase [deg]')
            ax.set_title(f'{_OUT_LABELS[col]} \u2190 {_IN_LABELS[col]}  (phase)', fontsize=9)

        # Print: diagonal peaks
        mask = _FREQS_HZ > 2.0
        peaks = []
        for i in range(3):
            diag_db  = mag_ct[:, i, i]
            idx_peak = np.argmax(diag_db[mask])
            peaks.append((float(_FREQS_HZ[mask][idx_peak]), float(diag_db[mask][idx_peak])))

        print(f'  {Y_val:>8.2f}  '
              f'{peaks[0][0]:>10.1f} / {peaks[0][1]:>7.1f}   '
              f'{peaks[1][0]:>10.1f} / {peaks[1][1]:>7.1f}   '
              f'{peaks[2][0]:>10.1f} / {peaks[2][1]:>7.1f}')

    print()
    # Frequency shift summary (diagonal channels)
    _print_freq_shift_summary(H_ct_cache)

    # Add colorbar-style legend + note for dashed
    axes[0, 2].legend(fontsize=7, loc='upper right')
    axes[0, 2].annotate('dashed = MATLAB ZOH DT', xy=(0.02, 0.04),
                         xycoords='axes fraction', fontsize=7, color='gray')
    plt.tight_layout()
    return fig


def _print_freq_shift_summary(H_ct_cache: dict):
    """Print diagonal channel peak frequency range across all Y values in _Y_FAMILY.

    Parameters
    ----------
    H_ct_cache : dict mapping Y_val (float) -> H_ct (nf, 3, 3) — precomputed FRFs.
    """
    print('  Diagonal peak frequency range across Y family (CT, min/max over all Y):')
    print(f"  {'Channel':14s}  {'f_min [Hz]':>12s}  {'f_max [Hz]':>12s}  "
          f"{'Range [Hz]':>12s}  {'Note':s}")
    print('  ' + '-' * 74)
    mask = _FREQS_HZ > 2.0
    for i, ch in enumerate(['X1<-F_X1', 'X2<-F_X2', 'Y<-F_Y']):
        peak_freqs = []
        for Y_val in _Y_FAMILY:
            mag  = _mag_db(H_ct_cache[Y_val])
            diag = mag[:, i, i]
            f_pk = float(_FREQS_HZ[mask][np.argmax(diag[mask])])
            peak_freqs.append(f_pk)
        f_min, f_max = min(peak_freqs), max(peak_freqs)
        rng  = f_max - f_min
        note = '(Y^2 symmetric: Y=0 gives max freq)' if i < 2 else '(pure inertia, no resonance peak)'
        print(f'  {ch:14s}  {f_min:>12.2f}  {f_max:>12.2f}  {rng:>12.2f}  {note}')
    print()


# ---------------------------------------------------------------------------
# Section 4: Natural frequencies vs Y
# ---------------------------------------------------------------------------

def _section_nat_freqs_vs_Y():
    """
    Section 4: Natural frequencies vs Y.

    Plots all oscillatory natural frequencies of A_c(Y) over Y ∈ [-0.4, +0.4] m.
    Shows how resonant frequencies shift with payload position — key motivation
    for LPV scheduling over a frozen LTI.
    """

    def _freqs_at_Y(Y_val: float) -> np.ndarray:
        """
        Natural frequencies [Hz] from A_c(Y) eigenvalues, sorted ascending.
        Returns one frequency per conjugate pair (positive imaginary part only).
        Result is zero-padded to 3 entries for consistent stacking.
        """
        Y_t    = torch.tensor(Y_val, dtype=_DTYPE)
        M_Y    = build_M(Y_t)
        eye3   = torch.eye(3, dtype=_DTYPE)
        MYinvK = torch.linalg.solve(M_Y, K).numpy()
        MYinvC = torch.linalg.solve(M_Y, C).numpy()
        A_c    = np.block([[np.zeros((3, 3)), np.eye(3)],
                           [-MYinvK,          -MYinvC  ]])
        eigs      = np.linalg.eigvals(A_c)
        pos_imag  = eigs.imag[eigs.imag > 1e-4]          # one per conjugate pair
        freqs_hz  = np.sort(np.abs(pos_imag) / (2 * np.pi))   # ascending Hz
        n = len(freqs_hz)
        if n >= 3:
            return freqs_hz[-3:]                          # take 3 highest
        return np.concatenate([np.zeros(3 - n), freqs_hz])   # zero-pad low end

    # Build (n_Y, 3) matrix across the full sweep
    freq_all = np.array([_freqs_at_Y(y) for y in _Y_SWEEP])   # (200, 3)

    # --- Print ---
    print(_SEP)
    print('[4/5]  NATURAL FREQUENCIES vs Y  -  A_c(Y) eigenvalue imaginary parts')
    print(_SEP)
    print(f'  Y sweep       : [{_Y_SWEEP[0]:.2f}, {_Y_SWEEP[-1]:.2f}] m  ({len(_Y_SWEEP)} points)')
    print(f'  Operating pt  : Y = {_Y_FREEZE:.3f} m')
    print()

    mode_labels = ['f1 (lowest)', 'f2', 'f3 (highest)']
    key_Y_vals  = [-0.40, 0.00, 0.30, 0.40]

    header = f"  {'Y [m]':>8s}  " + '  '.join([f'{lb:>18s}' for lb in mode_labels])
    print(header)
    print('  ' + '-' * 70)
    for Y_k in key_Y_vals:
        f     = _freqs_at_Y(Y_k)
        cells = []
        for fi in f:
            cells.append(f'{"(non-osc.)":>15s}' if fi < 0.1 else f'{fi:>13.2f} Hz')
        note = '  <- operating pt' if abs(Y_k - 0.30) < 0.01 else ''
        print(f'  {Y_k:>8.2f}  ' + '  '.join(cells) + note)
    print()

    print('  Range per mode across full Y sweep:')
    for i, label in enumerate(mode_labels):
        col = freq_all[:, i]
        osc = col[col > 0.1]
        if len(osc) == 0:
            print(f'  {label:14s}  (non-oscillatory)')
        else:
            print(f'  {label:14s}  {osc.min():.2f} - {osc.max():.2f} Hz  '
                  f'(delta {osc.max() - osc.min():.2f} Hz)')
    print()

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    fig.suptitle('Natural Frequencies vs Payload Position Y  '
                 '(A_c(Y) eigenvalue imaginary parts)', fontsize=12)

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (label, col) in enumerate(zip(mode_labels, colors)):
        f_col = freq_all[:, i]
        mask  = f_col > 0.1    # only plot oscillatory segments
        if mask.any():
            ax.plot(_Y_SWEEP[mask], f_col[mask], color=col, linewidth=2.0, label=label)

    ax.axvline(_Y_FREEZE, color='gray', linestyle='--', linewidth=1.2,
               label=f'Operating pt  Y = {_Y_FREEZE:.2f} m')
    ax.axvline(0.0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Payload position Y [m]', fontsize=11)
    ax.set_ylabel('Natural frequency [Hz]', fontsize=11)
    ax.set_xlim([_Y_SWEEP[0], _Y_SWEEP[-1]])
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Section 5: LPV vs frozen LTI trajectory comparison
# ---------------------------------------------------------------------------

def _bfr(y_ref: np.ndarray, y_hat: np.ndarray) -> float:
    """Best Fit Rate [%] = max(1 - ||y-yhat||2 / ||y-mean(y)||2, 0) * 100."""
    num = np.linalg.norm(y_ref - y_hat)
    den = np.linalg.norm(y_ref - y_ref.mean())
    if den < 1e-300:
        return 100.0 if num < 1e-300 else 0.0
    return float(max(1.0 - num / den, 0.0) * 100.0)


def _section_lpv_vs_frozen():
    """
    Section 5: LPV vs frozen LTI trajectory comparison.

    Both Python models are driven open-loop by u_q1 from MATLAB (closed-loop
    forces from the q1 path). Reference is q1 (MATLAB CT quasi-LPV, ode45).
    Frozen LTI: same physics as LPV but M(Y) frozen at Y=0.3 m.

    Prints BFR, RMSE, Max|error|, Mean|error| per model per channel.
    Figure: 4 panels — Y(t), positions, |LPV error| log, |frozen error| log.
    """
    path = os.path.join(_MAT_BASE, 'lpv_sim_varying_y.mat')
    if not os.path.exists(path):
        print(f'  SKIPPED - file not found: {path}')
        return None

    mat          = loadmat(path)
    q1_ref       = torch.tensor(mat['q1'],        dtype=_DTYPE)   # (N, 3) stage
    u_seq_stage  = torch.tensor(mat['u_q1'],      dtype=_DTYPE)   # (N, 3) stage forces
    u_q_stage    = torch.tensor(mat['u_q'],       dtype=_DTYPE) if 'u_q' in mat else None
    t_sim        = mat['t_sim'].squeeze()                          # (N,)
    q_simscape   = mat['q_simscape'] if 'q_simscape' in mat else None
    N            = q1_ref.shape[0]

    # Initial state: Y=0.3 m, all else zero (logical coords)
    x0 = torch.zeros(1, 6, dtype=_DTYPE)
    x0[0, 2] = 0.3

    u_batch = u_seq_stage.unsqueeze(0)   # (1, N, 3)

    with torch.no_grad():
        res_lpv    = simulate(
            x0, u_batch, _G_TRUE, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2, P, ts,
        )
        res_frozen = simulate_frozen(
            x0, u_batch, _G_TRUE, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2, P, ts,
            Y_freeze=_Y_FREEZE,
        )
        if u_q_stage is not None:
            u_q_batch      = u_q_stage.unsqueeze(0)
            res_lpv_uq     = simulate(
                x0, u_q_batch, _G_TRUE, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2, P, ts,
            )
            res_frozen_uq  = simulate_frozen(
                x0, u_q_batch, _G_TRUE, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2, P, ts,
                Y_freeze=_Y_FREEZE,
            )

    # Stage-coordinate outputs: (N, 3)
    y_lpv    = res_lpv.Y[0].numpy()      # Python LPV
    y_frozen = res_frozen.Y[0].numpy()   # Python frozen LTI
    y_ref    = q1_ref.numpy()            # MATLAB q1 (reference)

    err_lpv    = np.abs(y_lpv    - y_ref)
    err_frozen = np.abs(y_frozen - y_ref)

    # Y(t) scheduling variable from LPV state
    Y_traj = res_lpv.X[0, :N, 2].numpy()   # logical x[:,2] = Y position

    # --- Print ---
    print(_SEP)
    print('[5/5]  LPV vs FROZEN LTI  -  open-loop replay of u_q1')
    print(_SEP)
    print(f'  Source    : lpv_sim_varying_y.mat')
    print(f'  Steps     : {N}       Duration : {N / _FS:.3f} s')
    print(f'  Y range   : {Y_traj.min():.3f} to {Y_traj.max():.3f} m')
    print(f'  Frozen at : Y = {_Y_FREEZE:.3f} m')
    print()

    col_w = 14
    hdr   = (f"  {'Model':<20s}  {'Channel':<8s}  "
             f"{'BFR [%]':>{col_w}}  {'RMSE [m]':>{col_w}}  "
             f"{'Max|e| [m]':>{col_w}}  {'Mean|e| [m]':>{col_w}}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))

    for model_name, y_hat, err in [('Python LPV',    y_lpv,    err_lpv),
                                    ('Python frozen', y_frozen, err_frozen)]:
        for i, ch in enumerate(_CH_NAMES):
            bfr  = _bfr(y_ref[:, i], y_hat[:, i])
            rmse = float(np.sqrt(np.mean((y_ref[:, i] - y_hat[:, i])**2)))
            maxe = float(err[:, i].max())
            mne  = float(err[:, i].mean())
            print(f"  {model_name:<20s}  {ch:<8s}  "
                  f"{bfr:{col_w}.4f}  {rmse:{col_w}.3e}  "
                  f"{maxe:{col_w}.3e}  {mne:{col_w}.3e}")
        print()

    # Simscape secondary reference — driven by matched u_q forces
    if q_simscape is not None and u_q_stage is not None:
        q_sc = q_simscape.astype(np.float64)
        y_lpv_uq    = res_lpv_uq.Y[0].numpy()
        y_frozen_uq = res_frozen_uq.Y[0].numpy()
        print('  Secondary reference: Simscape  (driven by u_q -- matched forces)')
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for model_name, y_hat in [('Python LPV',    y_lpv_uq),
                                   ('Python frozen', y_frozen_uq)]:
            err_sc = np.abs(y_hat - q_sc)
            for i, ch in enumerate(_CH_NAMES):
                bfr  = _bfr(q_sc[:, i], y_hat[:, i])
                rmse = float(np.sqrt(np.mean((q_sc[:, i] - y_hat[:, i])**2)))
                maxe = float(err_sc[:, i].max())
                mne  = float(err_sc[:, i].mean())
                print(f"  {model_name:<20s}  {ch:<8s}  "
                      f"{bfr:{col_w}.4f}  {rmse:{col_w}.3e}  "
                      f"{maxe:{col_w}.3e}  {mne:{col_w}.3e}")
            print()

    # --- Settling error analysis ---
    # Detect settled regions: where |dY/dt| < threshold for at least min_hold samples.
    # This is a non-standard metric (the LPV-LFR literature uses full-trajectory BFR),
    # included as an application-specific diagnostic for the ASMPT gantry where
    # positioning accuracy after a move is the key performance indicator.
    _SETTLE_THRESH = 0.01   # m/s — below this, Y is considered stationary
    _SETTLE_HOLD   = 100    # samples at 20 kHz = 5 ms minimum hold time

    dY_dt = np.gradient(y_ref[:, 2], t_sim[:N])          # dY/dt from reference
    is_settled = np.abs(dY_dt) < _SETTLE_THRESH           # (N,) bool

    # Enforce minimum hold: erode short settled blips
    settled_mask = np.zeros(N, dtype=bool)
    run_start = None
    for k in range(N):
        if is_settled[k]:
            if run_start is None:
                run_start = k
        else:
            if run_start is not None and (k - run_start) >= _SETTLE_HOLD:
                settled_mask[run_start:k] = True
            run_start = None
    # Close final run
    if run_start is not None and (N - run_start) >= _SETTLE_HOLD:
        settled_mask[run_start:N] = True

    n_settled = settled_mask.sum()
    moving_mask = ~settled_mask

    if n_settled > 0:
        # Find contiguous settled regions for reporting
        changes     = np.diff(settled_mask.astype(np.int8))
        reg_starts  = np.where(changes == 1)[0] + 1
        reg_ends    = np.where(changes == -1)[0] + 1
        # Handle edge cases: settled at start/end
        if settled_mask[0]:
            reg_starts = np.concatenate([[0], reg_starts])
        if settled_mask[-1]:
            reg_ends = np.concatenate([reg_ends, [N]])

        print(f'  Settling analysis  (|dY/dt| < {_SETTLE_THRESH} m/s, hold >= {_SETTLE_HOLD} samples)')
        print(f'  Settled samples   : {n_settled} / {N} ({100 * n_settled / N:.1f}%)')
        for r, (rs, re) in enumerate(zip(reg_starts, reg_ends)):
            Y_at = y_ref[rs, 2]
            print(f'    Region {r}: t = {t_sim[rs]:.4f} - {t_sim[min(re-1, N-1)]:.4f} s  '
                  f'({re - rs} samples, {(re - rs) * _TS * 1e3:.1f} ms)  Y ~ {Y_at:.3f} m')
        print()

        settle_hdr = (f"  {'Model':<20s}  {'Channel':<8s}  "
                      f"{'RMSE settled':>14s}  {'Max|e| settled':>14s}  "
                      f"{'RMSE moving':>14s}  {'Max|e| moving':>14s}")
        print(settle_hdr)
        print('  ' + '-' * (len(settle_hdr) - 2))
        for model_name, y_hat in [('Python LPV', y_lpv), ('Python frozen', y_frozen)]:
            err_all = np.abs(y_hat - y_ref)
            for i, ch in enumerate(_CH_NAMES):
                e_s = err_all[settled_mask, i]
                e_m = err_all[moving_mask, i]
                rmse_s = float(np.sqrt(np.mean(e_s ** 2)))
                maxe_s = float(e_s.max())
                rmse_m = float(np.sqrt(np.mean(e_m ** 2))) if e_m.size > 0 else 0.0
                maxe_m = float(e_m.max()) if e_m.size > 0 else 0.0
                print(f"  {model_name:<20s}  {ch:<8s}  "
                      f"{rmse_s:14.3e}  {maxe_s:14.3e}  "
                      f"{rmse_m:14.3e}  {maxe_m:14.3e}")
            print()
    else:
        print(f'  Settling analysis: no settled regions detected '
              f'(threshold {_SETTLE_THRESH} m/s, hold {_SETTLE_HOLD} samples)')
        print()

    # --- Plot ---
    n_panels = 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 12), sharex=True)
    fig.suptitle('LPV vs Frozen LTI  |  open-loop replay of u_q1  |  ref = MATLAB q1',
                 fontsize=12)

    # Panel 1: Y(t) scheduling variable
    ax = axes[0]
    ax.plot(t_sim[:N], y_ref[:, 2],  color='tab:orange', linestyle='--',
            linewidth=1.5, label='q1 Y [ref]')
    ax.plot(t_sim[:N], Y_traj,       color='tab:blue',   linewidth=1.0,
            label='Python LPV Y', alpha=0.85)
    ax.axhline(_Y_FREEZE, color='gray', linestyle=':', linewidth=1.0,
               label=f'Frozen at {_Y_FREEZE:.2f} m')
    ax.set_ylabel('Y [m]')
    ax.set_title('Scheduling variable Y(t)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

    # Shade settled regions on all panels
    if n_settled > 0:
        for ax_i in axes:
            for rs, re in zip(reg_starts, reg_ends):
                ax_i.axvspan(t_sim[rs], t_sim[min(re - 1, N - 1)],
                             alpha=0.08, color='green', zorder=0)
        # Label once on Panel 1
        axes[0].annotate('green = settled', xy=(0.01, 0.04),
                          xycoords='axes fraction', fontsize=7, color='green')

    # Panel 2: position outputs — all 3 channels
    ax = axes[1]
    ch_colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i, (ch, col) in enumerate(zip(['X1', 'X2', 'Y'], ch_colors)):
        ax.plot(t_sim[:N], y_ref[:, i],    color=col, linestyle='--',
                linewidth=1.5, label=f'q1 {ch}',         alpha=0.9)
        ax.plot(t_sim[:N], y_lpv[:, i],    color=col, linestyle='-',
                linewidth=1.0, label=f'LPV {ch}',         alpha=0.85)
        ax.plot(t_sim[:N], y_frozen[:, i], color=col, linestyle=':',
                linewidth=1.2, label=f'Frozen {ch}',      alpha=0.7)
    ax.set_ylabel('Position [m]')
    ax.set_title('Position outputs  (-- q1 ref  |  solid LPV  |  dotted frozen)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right', ncol=3)

    # Panel 3: |error| LPV vs q1 — log scale
    ax = axes[2]
    for i, (ch, col) in enumerate(zip(['X1', 'X2', 'Y'], ch_colors)):
        ax.semilogy(t_sim[:N], err_lpv[:, i] + 1e-20, color=col,
                    linewidth=1.2, label=ch)
    ax.set_ylabel('|error| [m]')
    ax.set_title('|Python LPV - q1|  per channel')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    # Panel 4: |error| frozen vs q1 — log scale
    ax = axes[3]
    for i, (ch, col) in enumerate(zip(['X1', 'X2', 'Y'], ch_colors)):
        ax.semilogy(t_sim[:N], err_frozen[:, i] + 1e-20, color=col,
                    linewidth=1.2, label=ch)
    ax.set_ylabel('|error| [m]')
    ax.set_title('|Python frozen LTI - q1|  per channel')
    ax.set_xlabel('Time [s]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print()
    _section_trajectory()
    _section_bode_fixed_y()
    _section_bode_varying_y()
    _section_nat_freqs_vs_Y()
    _section_lpv_vs_frozen()

    print(_SEP)
    print('All 5 plots generated. Displaying...')
    print(_SEP)
    plt.show()
