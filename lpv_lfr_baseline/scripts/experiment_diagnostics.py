"""
experiment_diagnostics.py
-------------------------
Segment length diagnostic for the dual-gantry parameter recovery dataset.

Determines the BPTT segment length from a data-driven f_osc_min estimate:

    segment_len = ceil(N_PERIODS x fs_new / f_osc_min)

where f_osc_min is the lowest peak in the Welch PSD of the differential
(tilt) channel (X1 - X2), estimated from anti-symmetric trajectories.

    THEORY   : P&S Ch.2 §2.2.3 — Δf = fs_new/segment_len ≤ f_osc_min/N_PERIODS
               → segment_len ≥ N_PERIODS × fs_new / f_osc_min
    HEURISTIC: N_PERIODS = 3

Public API
----------
recommend_segment_len(trajs, fs_new, save_dir, dtype) -> int
    Called by precompute._compute(). Returns segment_len in samples at fs_new.

Run standalone:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics --dataset identification
"""

import math
import os

import matplotlib
matplotlib.use('Agg')   # non-interactive — safe on servers, always saves to file
import matplotlib.pyplot as plt
import torch
from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock, _build_matrices, _Lb,
)
from lpv_lfr_baseline.core.physics import build_M

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

_Y_OP_POINTS    = (0.00, 0.20, 0.30)       # frozen Y operating points [m]
_FS_CANDIDATES  = (1000, 2000, 4000, 8000)  # candidate new sampling rates [Hz]
# THEORY: Lecture 9 slides 10-12 (5SMB0) — "10 * omega_b <= omega_s <= 30 * omega_b"
_FS_RULE_FACTOR = 10                        # require fs_new >= factor * f_osc_min
# HEURISTIC: N_PERIODS = 3 — covers slowest mode with margin
N_PERIODS       = 3


# ----------------------------------------------------------------------
# fs_new / D helper  (used by precompute to determine decimation factor)
# ----------------------------------------------------------------------

def _get_f_osc_min(dtype=torch.float64):
    """
    Return slowest oscillatory mode frequency [Hz] from physics model.

    Computes eigenvalues of A_c = [[0, I], [-M(Y)^-1 K, -M(Y)^-1 C]] at
    each Y in _Y_OP_POINTS. Returns the minimum oscillatory frequency across
    all operating points. Used to select fs_new via the Lecture 9 rule:
        fs_new >= _FS_RULE_FACTOR * f_osc_min
    """
    block = ParameterizedLFRBlock(RMSE_baseline=1.0)
    Lb    = _Lb.to(dtype)
    with torch.no_grad():
        p = block._recover_params().to(dtype)
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = [p[i] for i in range(14)]
        params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
        _, _, _, K, C = _build_matrices(params_10, Lb, d)
        freqs_rads = []
        for Y_val in _Y_OP_POINTS:
            M_Y = build_M(torch.tensor(Y_val, dtype=dtype)).to(dtype)
            A_c = torch.zeros(6, 6, dtype=dtype)
            A_c[:3, 3:] = torch.eye(3, dtype=dtype)
            A_c[3:, :3] = -torch.linalg.solve(M_Y, K)
            A_c[3:, 3:] = -torch.linalg.solve(M_Y, C)
            osc = torch.linalg.eigvals(A_c).imag.abs()
            freqs_rads.extend(osc[osc > 1.0].tolist())
    return min(freqs_rads) / (2 * math.pi) if freqs_rads else 5.0


# ----------------------------------------------------------------------
# Segment length — Welch PSD of differential channel
# ----------------------------------------------------------------------

def _diag_segment_welch(trajs, fs_new, save_dir, f_osc_min_pole=None):
    """
    Estimate f_osc_min from Welch PSD of the differential (tilt) channel and
    derive the BPTT segment length.

    The differential signal diff = q1[:,0] - q1[:,1] (X1 - X2 stage position)
    is proportional to beam tilt. The tilt resonance (kb1+kb2 over effective
    tilt inertia, ~5 Hz) appears as a clear spectral peak in this channel.

    Anti-symmetric trajectories (id contains 'anti') are preferred because
    they inject energy directly into the diff mode, producing a sharper peak.
    Falls back to all trajectories if none contain 'anti'.

    Segment length derivation:
        THEORY: P&S Ch.2 §2.2.3 — leakage-free integer periods
            Δf = fs_new / segment_len ≤ f_osc_min / N_PERIODS
            → segment_len ≥ N_PERIODS × fs_new / f_osc_min
        HEURISTIC: N_PERIODS = 3 — covers slowest mode with margin

    Parameters
    ----------
    trajs          : list of traj dicts (decimated, at fs_new) — keys: id, q1
    fs_new         : effective sampling frequency [Hz] after decimation
    save_dir       : directory for plots (None = no plots)
    f_osc_min_pole : f_osc_min from pole analysis [Hz] (cross-check, informational)

    Returns
    -------
    dict: f_osc_min_FFT [Hz], segment_len [samples at fs_new], segment_len_s [s]
    """
    from scipy.signal import welch as _welch, find_peaks as _find_peaks

    # Prefer anti-symmetric trajectories — diff mode injected explicitly there
    anti_trajs = [t for t in trajs if 'anti' in t['id'].lower()]
    sel_trajs  = anti_trajs if anti_trajs else trajs

    print('\nSegment Length — data-driven f_osc_min from Welch PSD')
    print(f'  Source: {"anti-symmetric" if anti_trajs else "all"} trajectories — '
          f'{[t["id"] for t in sel_trajs]}')
    print(f'  Channel: diff = q1[:,0] - q1[:,1]  (X1 - X2 ∝ beam tilt)')

    # Frequency resolution: 3 bins per oscillatory period
    ref_f   = f_osc_min_pole if f_osc_min_pole is not None else 0.5
    delta_f = ref_f / 3.0
    nperseg = max(512, int(fs_new / delta_f))
    print(f'  Welch: delta_f={delta_f:.3f} Hz  nperseg={nperseg}'
          f'  ({fs_new / nperseg:.3f} Hz resolution)')

    psd_data = []
    with torch.no_grad():
        for traj in sel_trajs:
            diff   = (traj['q1'][:, 0] - traj['q1'][:, 1]).numpy()
            diff_c = diff - diff.mean()   # remove DC operating-point offset
            freqs_np, psd_np = _welch(diff_c, fs=fs_new, nperseg=nperseg, window='hann')
            psd_data.append((freqs_np, psd_np))
            print(f'  {traj["id"]}: diff RMS = {float(diff_c.std()):.4e} m')

    # Average PSD across selected trajectories
    psd_mean = sum(p for _, p in psd_data) / len(psd_data)
    freqs_np = psd_data[0][0]

    # Find peaks above 0.5 Hz (exclude DC and near-DC bins)
    min_idx      = max(1, int(0.5 / (freqs_np[1] - freqs_np[0])))
    psd_hi       = psd_mean[min_idx:]
    peaks_rel, _ = _find_peaks(psd_hi, prominence=psd_hi.max() * 0.01)
    peaks_abs    = peaks_rel + min_idx

    if peaks_rel.size > 0:
        f_osc_min_fft = float(freqs_np[peaks_abs[0]])   # lowest prominent peak
        print(f'  Peaks detected (lowest 5): '
              + ', '.join(f'{freqs_np[p]:.2f} Hz' for p in peaks_abs[:5]))
        print(f'  f_osc_min (FFT)  = {f_osc_min_fft:.3f} Hz  (lowest peak in diff Welch PSD)')
    else:
        # Fallback to pole-based estimate
        f_osc_min_fft = f_osc_min_pole if f_osc_min_pole is not None else 5.0
        print(f'  WARNING: no peaks found above 0.5 Hz in diff PSD.')
        print(f'  Falling back to pole-based estimate: {f_osc_min_fft:.3f} Hz')

    if f_osc_min_pole is not None:
        print(f'  Cross-check — pole analysis: {f_osc_min_pole:.3f} Hz  |  '
              f'FFT (data-driven): {f_osc_min_fft:.3f} Hz  |  '
              f'ratio: {f_osc_min_fft / f_osc_min_pole:.3f}')

    # THEORY: P&S Ch.2 §2.2.3 — Δf ≤ f_osc_min / N_PERIODS → segment_len ≥ N_PERIODS × fs_new / f_osc_min
    # HEURISTIC: N_PERIODS = 3
    segment_len   = math.ceil(N_PERIODS * fs_new / f_osc_min_fft)
    segment_len_s = segment_len / fs_new

    print(f'\n  segment_len = ceil(N_PERIODS x fs_new / f_osc_min_FFT)')
    print(f'             = ceil({N_PERIODS} x {fs_new:.0f} / {f_osc_min_fft:.3f})')
    print(f'             = {segment_len} samples  ({segment_len_s:.3f} s at {fs_new:.0f} Hz)')
    print(f'  [THEORY: P&S Ch.2 §2.2.3 — Δf ≤ f_osc_min / N_PERIODS → leakage-free integer periods]')
    print(f'  [HEURISTIC: N_PERIODS = {N_PERIODS}]')

    if save_dir is not None:
        _plot_segment_welch(
            freqs_np, psd_mean, psd_data, sel_trajs,
            peaks_abs, f_osc_min_fft, f_osc_min_pole,
            segment_len, segment_len_s, fs_new, save_dir,
        )

    return {
        'f_osc_min_FFT':  f_osc_min_fft,
        'f_osc_min_pole': f_osc_min_pole,
        'segment_len':    segment_len,
        'segment_len_s':  segment_len_s,
    }


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def recommend_segment_len(trajs, fs_new, save_dir, dtype=torch.float64):
    """
    Determine segment_len from data-driven FFT-based f_osc_min estimation.
    Called by precompute._compute() — prints, no plots.

    Uses Welch PSD of the differential channel (X1-X2) from anti-symmetric
    trajectories to find f_osc_min nonparametrically. The pole-based estimate
    from the physics model is passed as a cross-check reference only.

    Parameters
    ----------
    trajs    : list of traj dicts (decimated, at fs_new) — keys: id, q1
    fs_new   : target sampling frequency [Hz] (after decimation)
    save_dir : directory for cache artefacts (unused here, passed for consistency)
    dtype    : torch dtype (default float64; kept for API compatibility)

    Returns
    -------
    int - segment_len in samples at fs_new
    """
    f_osc_min_pole = _get_f_osc_min()
    r_seg = _diag_segment_welch(trajs, fs_new, save_dir=None,
                                f_osc_min_pole=f_osc_min_pole)
    return r_seg['segment_len']


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _plot_segment_welch(freqs_np, psd_mean, psd_data, sel_trajs,
                        peaks_abs, f_osc_min_fft, f_osc_min_pole,
                        segment_len, segment_len_s, fs_new, save_dir):
    """Plot Welch PSD of differential channel with f_osc_min annotation and segment derivation."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Top: Welch PSD of diff channel
    ax = axes[0]
    for (freqs_t, psd_t), traj in zip(psd_data, sel_trajs):
        ax.semilogy(freqs_t[1:], psd_t[1:], alpha=0.55, linewidth=0.9, label=traj['id'])
    ax.semilogy(freqs_np[1:], psd_mean[1:], 'k-', linewidth=1.8, label='Mean PSD', zorder=5)
    ax.axvline(f_osc_min_fft, color='tab:red', linestyle='--', linewidth=1.6,
               label=f'f_osc_min FFT = {f_osc_min_fft:.2f} Hz')
    if f_osc_min_pole is not None:
        ax.axvline(f_osc_min_pole, color='tab:blue', linestyle=':', linewidth=1.5,
                   label=f'f_osc_min pole = {f_osc_min_pole:.2f} Hz')
    for p in peaks_abs[:8]:
        ax.axvline(freqs_np[p], color='tab:orange', alpha=0.35, linewidth=0.7)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [m²/Hz]')
    ax.set_title('Welch PSD — differential channel (X1−X2 ∝ beam tilt)\n'
                 'Peak = tilt resonance (kb1+kb2 / effective tilt inertia)')
    ax.legend(fontsize=8, ncol=3, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    xlim_max = min(fs_new / 2, max(20 * f_osc_min_fft, 50.0))
    ax.set_xlim(left=max(0.2, freqs_np[1]), right=xlim_max)

    # Bottom: segment length derivation as annotated text
    ax2 = axes[1]
    ax2.axis('off')
    pole_line = (f'  f_osc_min (poles) = {f_osc_min_pole:.3f} Hz\n'
                 if f_osc_min_pole is not None else '')
    text = (
        'Segment length derivation\n'
        '\n'
        '  THEORY  : Δf = fs_new / segment_len ≤ f_osc_min / N_PERIODS\n'
        '            → segment_len ≥ N_PERIODS × fs_new / f_osc_min\n'
        '            [P&S Ch.2 §2.2.3 — leakage-free integer periods]\n'
        '\n'
        f'  HEURISTIC : N_PERIODS = {N_PERIODS}\n'
        '\n'
        f'  f_osc_min (FFT)   = {f_osc_min_fft:.3f} Hz\n'
        + pole_line +
        f'  fs_new            = {fs_new:.0f} Hz\n'
        '\n'
        f'  segment_len = ceil({N_PERIODS} x {fs_new:.0f} / {f_osc_min_fft:.3f})\n'
        f'             = {segment_len} samples  ({segment_len_s:.3f} s)'
    )
    ax2.text(0.03, 0.95, text, transform=ax2.transAxes,
             fontsize=9.5, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    _save_fig(fig, save_dir, 'diag_segment_welch.png')


def _save_fig(fig, save_dir, filename):
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot saved: {path}')


# ----------------------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    from lpv_lfr_baseline.scripts.train_param_recovery import _DATASETS, DATASET
    from lpv_lfr_baseline.scripts.precompute import (
        _load_trajectory, _build_state_traj_logical,
    )
    from lpv_lfr_baseline.core.physics import P as _P, ts as _ts

    parser = argparse.ArgumentParser(
        description='Segment length diagnostic for dual-gantry parameter recovery.'
    )
    parser.add_argument(
        '--dataset', default=DATASET, choices=list(_DATASETS),
        help='Dataset to analyse (default: active DATASET in train_param_recovery.py)',
    )
    args = parser.parse_args()

    ds       = _DATASETS[args.dataset]
    save_dir = os.path.join(ds['save_dir'], 'diagnostics')
    os.makedirs(save_dir, exist_ok=True)

    print(f'Dataset  : {args.dataset}')
    print(f'traj_dir : {ds["traj_dir"]}')
    print(f'save_dir : {save_dir}')

    _dtype = torch.float64
    _P_d   = _P.to(_dtype)
    _ts_d  = float(_ts)

    # --- Determine fs_new and D from physics model ---
    # THEORY: Lecture 9 slides 10-12 (5SMB0) — fs_new >= _FS_RULE_FACTOR * f_osc_min
    f_osc_min_pole = _get_f_osc_min()
    fs_new = next(
        (f for f in _FS_CANDIDATES if f >= _FS_RULE_FACTOR * f_osc_min_pole),
        None,
    )
    if fs_new is None:
        raise RuntimeError(f'No fs candidate >= {_FS_RULE_FACTOR} x {f_osc_min_pole:.2f} Hz '
                           f'in {_FS_CANDIDATES}')
    print(f'\nSampling rate selection:')
    print(f'  f_osc_min (poles) = {f_osc_min_pole:.4f} Hz')
    print(f'  rule: fs_new >= {_FS_RULE_FACTOR} x f_osc_min = {_FS_RULE_FACTOR * f_osc_min_pole:.1f} Hz')
    print(f'  -> fs_new = {fs_new} Hz')

    # --- Load and decimate trajectories ---
    print('\nLoading trajectories...')
    trajs = []
    for spec in ds['traj_specs']:
        mat_path = os.path.join(ds['traj_dir'], spec['file'])
        u, q1, fs_traj = _load_trajectory(mat_path, _dtype)
        fs_orig = float(fs_traj)
        D = round(fs_orig / fs_new)
        state_traj = _build_state_traj_logical(q1, _P_d, _ts_d, _dtype)
        trajs.append({
            'id':         spec['id'],
            'N':          int(q1[::D].shape[0]),
            'fs':         fs_new,
            'u':          u[:, ::D, :],
            'q1':         q1[::D, :],
            'state_traj': state_traj[::D, :],
        })
        print(f'  {spec["id"]}: T_orig={q1.shape[0]}  D={D}  T_dec={trajs[-1]["N"]}  fs_dec={fs_new} Hz')

    # --- Run segment length diagnostic with plots ---
    r_seg = _diag_segment_welch(trajs, fs_new, save_dir,
                                f_osc_min_pole=f_osc_min_pole)

    print()
    print('=' * 60)
    print('  SEGMENT LENGTH DIAGNOSTIC SUMMARY')
    print('-' * 60)
    print(f'  fs_new           : {fs_new} Hz')
    print(f'  f_osc_min (pole) : {f_osc_min_pole:.4f} Hz  [informational]')
    print(f'  f_osc_min (FFT)  : {r_seg["f_osc_min_FFT"]:.4f} Hz  [drives segment_len]')
    print(f'  segment_len      : {r_seg["segment_len"]} samples'
          f'  ({r_seg["segment_len_s"]:.3f} s at {fs_new} Hz)')
    print('=' * 60)
