"""
experiment_diagnostics.py
-------------------------
Experiment-level diagnostics for the dual-gantry parameter recovery dataset.

Diagnostics (run in order — each feeds the next)
-------------------------------------------------
1. FFT / frequency content   — sampling rate and decimation factor recommendation
2. Step response             — dominant time constant τ_max per Y operating point  [Prompt 2]
3. Parameter sensitivity     — minimum segment length per parameter                [Prompt 2]
4. Observability             — horizon sanity check (expected: 2 samples)          [Prompt 2]

Public API
----------
recommend_segment_len(trajs, fs, save_dir, dtype) -> int
    Called by precompute._compute(). Runs diagnostics 2+3 only (no plots).
    Returns segment_len in samples at the given fs.

run_all_diagnostics(pre, save_dir) -> None
    Runs all four diagnostics, saves plots, prints full summary.
    Called by __main__.

Run standalone:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics --dataset base
"""

import os

import matplotlib
matplotlib.use('Agg')   # non-interactive — safe on servers, always saves to file
import matplotlib.pyplot as plt
import torch

from lpv_lfr_baseline.blocks.lfr_param_block import _PARAM_NAMES

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

_Y_OP_POINTS       = (0.00, 0.20, 0.30)        # frozen Y operating points [m]
_T_SENS_MULTIPLIER = 10                         # T_sens = multiplier * τ_max
_ENERGY_THRESHOLD  = 0.95                       # cumulative sensitivity energy to capture
_FS_CANDIDATES        = (1000, 2000, 4000, 8000)   # candidate new sampling rates [Hz]
_FS_RULE_FACTOR       = 8                          # require fs_new >= factor * f_99
_FFT_ENERGY_THRESHOLD = 0.99                       # cumulative PSD energy for f_99

_PARAM_CATEGORIES = {
    'mh': 'inertial', 'm1': 'inertial', 'm2': 'inertial', 'mb': 'inertial',
    'Jb': 'inertial', 'Jh': 'inertial',
    'cg1': 'damping',  'cg2': 'damping', 'cy': 'damping',
    'cb1': 'damping',  'cb2': 'damping',
    'kb1': 'stiffness', 'kb2': 'stiffness',
    'd':  'geometry',
}

_CATEGORY_COLORS = {
    'inertial':  'steelblue',
    'damping':   'darkorange',
    'stiffness': 'forestgreen',
    'geometry':  'mediumpurple',
}

_CH_NAMES = ('X1', 'X2', 'Y')


# ----------------------------------------------------------------------
# Diagnostic 1 — FFT / frequency content
# ----------------------------------------------------------------------

def _diag_fft(trajs, save_dir):
    """
    Compute PSD of each output channel for all trajectories.

    Centres signals before FFT to remove the DC operating-point offset.
    Finds f_99: the frequency below which 99% of signal power lies per channel.
    Recommends the smallest fs_new in _FS_CANDIDATES satisfying
        fs_new >= _FS_RULE_FACTOR * f_99_overall.

    Returns
    -------
    dict: f99_overall [Hz], fs_new [Hz], decimation_factor
    """
    fs_orig = float(trajs[0]['fs'])

    f99_by_traj = []   # [[f99_X1, f99_X2, f99_Y], ...] one list per trajectory
    psd_data    = []   # [(freqs_np, psd_np), ...] for plotting — DC bin excluded

    with torch.no_grad():
        for traj in trajs:
            q1  = traj['q1']                              # (T, 3)
            q1c = q1 - q1.mean(dim=0, keepdim=True)       # remove DC offset
            T   = q1c.shape[0]

            freqs = torch.fft.rfftfreq(T, d=1.0 / fs_orig)       # (T//2+1,)
            psd   = torch.fft.rfft(q1c, dim=0).abs().pow(2) / T  # (T//2+1, 3)

            cum = psd.cumsum(dim=0) / psd.sum(dim=0, keepdim=True).clamp(min=1e-30)
            f99s = []
            for c in range(3):
                hits = (cum[:, c] >= _FFT_ENERGY_THRESHOLD).nonzero(as_tuple=True)[0]
                f99s.append(float(freqs[hits[0]]) if hits.numel() > 0 else fs_orig / 2)
            f99_by_traj.append(f99s)
            psd_data.append((freqs[1:].numpy(), psd[1:].numpy()))  # skip DC bin

    f99_overall = max(f for f99s in f99_by_traj for f in f99s)
    fs_new = next(
        (f for f in _FS_CANDIDATES if f >= _FS_RULE_FACTOR * f99_overall),
        int(fs_orig),
    )
    D = round(fs_orig / fs_new)

    # --- Print ---
    print('\nFFT Analysis')
    print(f'  fs_original = {fs_orig:.0f} Hz')
    print(f'  {"traj":<6}' + ''.join(f'  {ch:>10}' for ch in _CH_NAMES))
    for traj, f99s in zip(trajs, f99_by_traj):
        print(f'  {traj["id"]:<6}' + ''.join(f'  {f:>9.1f}Hz' for f in f99s))
    print(f'  f_99 overall (max across all channels + trajectories): {f99_overall:.1f} Hz')
    print(f'  Rule: fs_new >= {_FS_RULE_FACTOR} x {f99_overall:.0f} = {_FS_RULE_FACTOR * f99_overall:.0f} Hz')
    print(f'  -> Recommended fs_new = {fs_new} Hz  (decimation factor D={D})')

    # --- Plot: 3 subplots (one per channel), all trajectories overlaid, log-log ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for c, (ax, ch) in enumerate(zip(axes, _CH_NAMES)):
        for (freqs_np, psd_np), traj in zip(psd_data, trajs):
            ax.loglog(freqs_np, psd_np[:, c], alpha=0.7, linewidth=0.9, label=traj['id'])
        ax.axvline(f99_overall, color='tab:red', linestyle='--', linewidth=1.4,
                   label=f'f_99 = {f99_overall:.0f} Hz' if c == 0 else '_')
        ax.axvline(fs_new / 2, color='tab:green', linestyle=':', linewidth=1.4,
                   label=f'Nyquist @ {fs_new} Hz' if c == 0 else '_')
        ax.set_ylabel(f'PSD {ch} [m2/sample]')
        ax.grid(True, which='both', alpha=0.3)
        if c == 0:
            ax.legend(fontsize=7, ncol=5, loc='upper right')
    axes[-1].set_xlabel('Frequency [Hz]')
    fig.suptitle(
        f'FFT  —  f_99={f99_overall:.0f} Hz  ->  Recommended fs={fs_new} Hz (D={D})',
        fontsize=11,
    )
    plt.tight_layout()
    _save_fig(fig, save_dir, 'diag_fft.png')

    return {'f99_overall': f99_overall, 'fs_new': fs_new, 'decimation_factor': D}


# ----------------------------------------------------------------------
# Diagnostics 2, 3, 4 — implemented in Prompt 2
# ----------------------------------------------------------------------

def _diag_step_response(fs, save_dir, dtype=torch.float64):
    """Frozen step response at _Y_OP_POINTS. Returns tau_max and pole info."""
    raise NotImplementedError('Prompt 2')


def _diag_param_sensitivity(trajs, fs, tau_max, save_dir, dtype=torch.float64):
    """
    Parameter sensitivity dY/d(log_params) over time via autograd.
    Returns segment_len_samples and t_95 per parameter.
    """
    raise NotImplementedError('Prompt 2')


def _diag_observability(fs, save_dir, dtype=torch.float64):
    """
    Observability matrix rank growth at _Y_OP_POINTS.
    Returns horizon (expected: 2 for C=[I, 0]).
    """
    raise NotImplementedError('Prompt 2')


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def recommend_segment_len(trajs, fs, save_dir, dtype=torch.float64):
    """
    Determine segment_len from step response + parameter sensitivity.
    Called by precompute._compute() — prints only, no plots.

    Parameters
    ----------
    trajs    : list of traj dicts from precompute (id, u, q1, state_traj, N, fs)
    fs       : sampling frequency [Hz] (native or decimated)
    save_dir : directory for cache / artefacts
    dtype    : torch dtype (default float64)

    Returns
    -------
    int — segment_len in samples at the given fs
    """
    raise NotImplementedError('Prompt 2')


def run_all_diagnostics(trajs, save_dir):
    """
    Run all four diagnostics. Saves plots to save_dir, prints full summary.

    Parameters
    ----------
    trajs    : list of traj dicts — each with keys: id, u, q1, state_traj, N, fs
    save_dir : directory to save plots (created if absent)
    """
    os.makedirs(save_dir, exist_ok=True)
    fs = float(trajs[0]['fs'])

    r_fft = _diag_fft(trajs, save_dir)

    # Prompt 2 — uncomment when implemented:
    # r_step = _diag_step_response(fs, save_dir)
    # r_sens = _diag_param_sensitivity(trajs, fs, r_step['tau_max'], save_dir)
    # r_obs  = _diag_observability(fs, save_dir)

    # Summary (partial until Prompt 2)
    print()
    print('=' * 52)
    print('  EXPERIMENT DIAGNOSTICS SUMMARY')
    print('-' * 52)
    print(f'  Recommended fs        : {r_fft["fs_new"]} Hz'
          f'  (D={r_fft["decimation_factor"]} from {fs:.0f} Hz)')
    print('  tau_max               : pending  (step response — Prompt 2)')
    print('  Slowest parameter     : pending  (param sensitivity — Prompt 2)')
    print('  segment_len           : pending')
    print('  Observability horizon : pending  (Prompt 2)')
    print('=' * 52)


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

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
        description='Experiment diagnostics for dual-gantry parameter recovery.'
    )
    parser.add_argument(
        '--dataset', default=DATASET, choices=list(_DATASETS),
        help='Dataset to analyse (default: active DATASET in train_param_recovery.py)',
    )
    args = parser.parse_args()

    ds       = _DATASETS[args.dataset]
    save_dir = os.path.join(ds['save_dir'], 'diagnostics')

    print(f'Dataset  : {args.dataset}')
    print(f'traj_dir : {ds["traj_dir"]}')
    print(f'save_dir : {save_dir}')

    _dtype = torch.float64
    _P_d   = _P.to(_dtype)
    _ts_d  = float(_ts)

    print('Loading trajectories...')
    trajs = []
    for spec in ds['traj_specs']:
        mat_path = os.path.join(ds['traj_dir'], spec['file'])
        u, q1, fs = _load_trajectory(mat_path, _dtype)
        state_traj = _build_state_traj_logical(q1, _P_d, _ts_d, _dtype)
        trajs.append({
            'id':         spec['id'],
            'N':          int(q1.shape[0]),
            'fs':         fs,
            'u':          u,
            'q1':         q1,
            'state_traj': state_traj,
        })
        print(f'  {spec["id"]}: T={q1.shape[0]}, fs={fs:.0f} Hz')

    run_all_diagnostics(trajs, save_dir)
