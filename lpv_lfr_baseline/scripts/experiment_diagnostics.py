"""
experiment_diagnostics.py
-------------------------
Experiment-level diagnostics for the dual-gantry parameter recovery dataset.

Diagnostics (run in order - each feeds the next)
-------------------------------------------------
1. FFT / frequency content   - sampling rate and decimation factor recommendation
2. Step response             - dominant time constant tau_max and segment length
3. Observability             - horizon sanity check (expected: 2 samples)

Public API
----------
recommend_segment_len(fs, fs_new, save_dir, dtype) -> int
    Called by precompute._compute(). Runs diagnostic 2 only (no plots).
    Returns segment_len in samples at fs_new.

run_all_diagnostics(trajs, save_dir) -> None
    Runs all three diagnostics, saves plots, prints full summary.
    Called by __main__.

Run standalone:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics --dataset base
"""

import math
import os

import matplotlib
matplotlib.use('Agg')   # non-interactive - safe on servers, always saves to file
import matplotlib.pyplot as plt
import torch
from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock, _build_matrices, _Lb, _PARAM_NAMES,
)
from lpv_lfr_baseline.core.physics import (
    build_poly_constants, build_M, P as _P, ts as _ts,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate_frozen

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

_Y_OP_POINTS          = (0.00, 0.20, 0.30)        # frozen Y operating points [m]
_FS_CANDIDATES        = (1000, 2000, 4000, 8000)   # candidate new sampling rates [Hz]
_FS_RULE_FACTOR       = 8                          # require fs_new >= factor * f_99
_FFT_ENERGY_THRESHOLD = 0.99                       # cumulative PSD energy for f_99
N_PERIODS             = 3                          # number of slowest oscillatory periods per segment

_CH_NAMES = ('X1', 'X2', 'Y')


# ----------------------------------------------------------------------
# Shared helper - differentiable matrix build from log_params
# ----------------------------------------------------------------------

def _build_sim_matrices(log_p, params_init, dtype):
    """
    Differentiably rebuild all simulation matrices from a log_params tensor.

    Replicates ParameterizedLFRBlock._recover_params() + _build_matrices() +
    build_poly_constants() + build_G_matrix() as a pure function so that
    torch.autograd.functional.jvp can differentiate through it.

    Parameters
    ----------
    log_p       : (14,) log-space parameter tensor (leaf, requires_grad optional)
    params_init : (14,) detuned initial physical values (constant buffer)
    dtype       : torch dtype

    Returns
    -------
    G, K, C, mh, alpha, beta, gamma, N0, N1, N2
    """
    Lb = _Lb.to(dtype)
    p  = (params_init * log_p.exp()).clamp(min=1e-6)
    kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = [p[i] for i in range(14)]
    params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
    _, M1, M2, K, C = _build_matrices(params_10, Lb, d)
    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(m1, m2, mb, mh, Jb, Jh, Lb, d)
    d0 = mh * (alpha * gamma - beta ** 2)
    G  = build_G_matrix(N0, d0, M1, M2, K, C)
    return G, K, C, mh, alpha, beta, gamma, N0, N1, N2


# ----------------------------------------------------------------------
# Diagnostic 1 - FFT / frequency content
# ----------------------------------------------------------------------

def _diag_fft(trajs, save_dir):
    """
    Compute PSD of each output channel for all trajectories.

    Centres signals before FFT to remove the DC operating-point offset.
    Finds f_99: the frequency below which _FFT_ENERGY_THRESHOLD of signal power
    lies per channel. Recommends the smallest fs_new in _FS_CANDIDATES satisfying
        fs_new >= _FS_RULE_FACTOR * f_99_overall.

    Returns
    -------
    dict: f99_overall [Hz], fs_new [Hz], decimation_factor
    """
    fs_orig = float(trajs[0]['fs'])

    f99_by_traj = []
    psd_data    = []

    with torch.no_grad():
        for traj in trajs:
            q1  = traj['q1']
            q1c = q1 - q1.mean(dim=0, keepdim=True)
            T   = q1c.shape[0]

            freqs = torch.fft.rfftfreq(T, d=1.0 / fs_orig)
            psd   = torch.fft.rfft(q1c, dim=0).abs().pow(2) / T

            cum  = psd.cumsum(dim=0) / psd.sum(dim=0, keepdim=True).clamp(min=1e-30)
            f99s = []
            for c in range(3):
                hits = (cum[:, c] >= _FFT_ENERGY_THRESHOLD).nonzero(as_tuple=True)[0]
                f99s.append(float(freqs[hits[0]]) if hits.numel() > 0 else fs_orig / 2)
            f99_by_traj.append(f99s)
            psd_data.append((freqs[1:].numpy(), psd[1:].numpy()))

    f99_overall = max(f for f99s in f99_by_traj for f in f99s)
    fs_new = next(
        (f for f in _FS_CANDIDATES if f >= _FS_RULE_FACTOR * f99_overall),
        int(fs_orig),
    )
    D = round(fs_orig / fs_new)

    print('\nFFT Analysis')
    print(f'  fs_original = {fs_orig:.0f} Hz')
    print(f'  {"traj":<6}' + ''.join(f'  {ch:>10}' for ch in _CH_NAMES))
    for traj, f99s in zip(trajs, f99_by_traj):
        print(f'  {traj["id"]:<6}' + ''.join(f'  {f:>9.1f}Hz' for f in f99s))
    print(f'  f_99 overall (max across all channels + trajectories): {f99_overall:.1f} Hz')
    print(f'  Rule: fs_new >= {_FS_RULE_FACTOR} x {f99_overall:.0f} = {_FS_RULE_FACTOR * f99_overall:.0f} Hz')
    print(f'  -> Recommended fs_new = {fs_new} Hz  (decimation factor D={D})')

    if save_dir is not None:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for c, (ax, ch) in enumerate(zip(axes, _CH_NAMES)):
            for (freqs_np, psd_np), traj in zip(psd_data, trajs):
                ax.loglog(freqs_np, psd_np[:, c], alpha=0.7, linewidth=0.9, label=traj['id'])
            ax.axvline(f99_overall, color='tab:red', linestyle='--', linewidth=1.4,
                       label=f'f_99 = {f99_overall:.0f} Hz' if c == 0 else '_')
            ax.axvline(fs_new / 2, color='tab:green', linestyle=':', linewidth=1.4,
                       label=f'Nyquist @ {fs_new} Hz' if c == 0 else '_')
            ax.set_ylabel(f'PSD {ch} [m²/sample]')
            ax.grid(True, which='both', alpha=0.3)
            if c == 0:
                ax.legend(fontsize=7, ncol=5, loc='upper right')
        axes[-1].set_xlabel('Frequency [Hz]')
        fig.suptitle(
            f'FFT  -  f_99={f99_overall:.0f} Hz  ->  Recommended fs={fs_new} Hz (D={D})',
            fontsize=11,
        )
        plt.tight_layout()
        _save_fig(fig, save_dir, 'diag_fft.png')

    return {'f99_overall': f99_overall, 'fs_new': fs_new, 'decimation_factor': D,
            'f99_by_traj': f99_by_traj}


# ----------------------------------------------------------------------
# Diagnostic 2 - Step response / pole analysis
# ----------------------------------------------------------------------

def _diag_step_response(fs, fs_new, save_dir, dtype=torch.float64):
    """
    Compute eigenvalues of A_c = [[0, I], [-M(Y)^-1 K, -M(Y)^-1 C]] at each
    Y in _Y_OP_POINTS using detuned initial parameters.

    Time constants: tau_i = -1 / Re(lambda_i) for Re(lambda_i) < 0.
    tau_max = max over all poles and operating points.

    Oscillatory poles (|Im(lambda)| > 1.0 rad/s) determine the slowest
    oscillatory frequency f_osc_min and hence the segment length:
        segment_len_s = N_PERIODS / f_osc_min
        segment_len   = ceil(segment_len_s * fs_new)

    Returns
    -------
    dict: tau_max [s], poles {Y_val: eigvals tensor},
          f_osc_min [Hz], segment_len [samples at fs_new], segment_len_s [s]
    """
    block = ParameterizedLFRBlock(RMSE_baseline=1.0)
    Lb    = _Lb.to(dtype)

    with torch.no_grad():
        p = block._recover_params().to(dtype)
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = [p[i] for i in range(14)]
        params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
        _, _, _, K, C = _build_matrices(params_10, Lb, d)

    poles   = {}
    tau_max = 0.0

    print('\nStep Response / Pole Analysis')
    for Y_val in _Y_OP_POINTS:
        with torch.no_grad():
            M_Y = build_M(torch.tensor(Y_val, dtype=dtype)).to(dtype)
            A_c = torch.zeros(6, 6, dtype=dtype)
            A_c[:3, 3:] = torch.eye(3, dtype=dtype)
            A_c[3:, :3] = -torch.linalg.solve(M_Y, K)
            A_c[3:, 3:] = -torch.linalg.solve(M_Y, C)
            eigvals = torch.linalg.eigvals(A_c)   # (6,) complex

        poles[Y_val] = eigvals

        neg_real   = eigvals.real[eigvals.real < -1e-8]
        tau_max_Y  = float((-1.0 / neg_real).max()) if neg_real.numel() > 0 else 0.0
        tau_max    = max(tau_max, tau_max_Y)

        # Re≈0: rigid-body modes (no X1/X2 stiffness open-loop - expected)
        # Re>1e-4: genuinely unstable - warn
        n_rb       = int((eigvals.real.abs() < 1e-4).sum())
        n_unstable = int((eigvals.real > 1e-4).sum())
        note  = f'  ({n_rb} rigid-body)' if n_rb else ''
        warn  = f'  WARNING: {n_unstable} unstable pole(s)' if n_unstable else ''
        print(f'  Y={Y_val:.2f} m : tau_max={tau_max_Y:.4f} s'
              f'  poles = [{", ".join(f"{e.real.item():+.2f}{e.imag.item():+.2f}j" for e in eigvals)}]'
              + note + warn)

    print(f'  tau_max overall = {tau_max:.4f} s')

    # Oscillatory poles: |Im(lambda)| > 1.0 rad/s
    all_osc_freqs_rads = []
    for eigvals in poles.values():
        osc_mask = eigvals.imag.abs() > 1.0
        if osc_mask.any():
            all_osc_freqs_rads.extend(eigvals.imag[osc_mask].abs().tolist())

    f_osc_min     = min(all_osc_freqs_rads) / (2 * math.pi)
    segment_len_s = N_PERIODS / f_osc_min
    segment_len   = math.ceil(segment_len_s * fs_new)

    print(f'  f_osc_min = {f_osc_min:.4f} Hz  (slowest oscillatory mode)')
    print(f'  segment_len_s = {segment_len_s:.4f} s  ({N_PERIODS} periods of f_osc_min)')
    print(f'  segment_len = {segment_len} samples at {fs_new:.0f} Hz')

    if save_dir is not None:
        _plot_poles(poles, tau_max, save_dir)

    return {
        'tau_max':       tau_max,
        'poles':         poles,
        'f_osc_min':     f_osc_min,
        'segment_len':   segment_len,
        'segment_len_s': segment_len_s,
    }


# ----------------------------------------------------------------------
# Diagnostic 3 - Observability
# ----------------------------------------------------------------------

def _diag_observability(fs, save_dir, dtype=torch.float64):
    """
    Build observability matrix O_h = [C; CA_d; ...; CA_d^{h-1}] at each
    Y in _Y_OP_POINTS and track rank growth.

    C = [P^T | 0_{3x3}]  (3x6): output is y_stage = q_logical @ P, so
    in column-vector convention C selects first 3 states and applies P^T.

    Expected: rank reaches 6 at h=2 since
        [C; CA_c] = [[P^T, 0], [0, P^T]] - block diagonal of invertible P^T.

    Returns
    -------
    dict: horizon int, rank_profiles {Y_val: [rank_1, rank_2, ...]}
    """
    block = ParameterizedLFRBlock(RMSE_baseline=1.0)
    Lb    = _Lb.to(dtype)
    P     = _P.to(dtype)
    ts_s  = float(_ts)

    with torch.no_grad():
        p = block._recover_params().to(dtype)
        kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = [p[i] for i in range(14)]
        params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
        _, _, _, K, C_damp = _build_matrices(params_10, Lb, d)

    # Output matrix: y_stage = q_logical @ P  =>  C_obs = [P^T | 0] (3x6)
    C_obs = torch.cat([P.T, torch.zeros(3, 3, dtype=dtype)], dim=1)   # (3, 6)

    rank_profiles = {}
    horizon       = None

    print('\nObservability Analysis')
    for Y_val in _Y_OP_POINTS:
        with torch.no_grad():
            M_Y = build_M(torch.tensor(Y_val, dtype=dtype)).to(dtype)
            A_c = torch.zeros(6, 6, dtype=dtype)
            A_c[:3, 3:] = torch.eye(3, dtype=dtype)
            A_c[3:, :3] = -torch.linalg.solve(M_Y, K)
            A_c[3:, 3:] = -torch.linalg.solve(M_Y, C_damp)
            A_d = torch.linalg.matrix_exp(A_c * ts_s)

            ranks    = []
            h_full   = None
            A_pow    = torch.eye(6, dtype=dtype)
            O_rows   = [C_obs]

            for h in range(1, 9):
                O      = torch.cat(O_rows, dim=0)
                rank_h = int(torch.linalg.matrix_rank(O).item())
                ranks.append(rank_h)
                if rank_h == 6:
                    h_full = h
                    break
                A_pow = A_pow @ A_d
                O_rows.append(C_obs @ A_pow)

        rank_profiles[Y_val] = ranks
        h_str = str(h_full) if h_full is not None else f'>{h}'
        print(f'  Y={Y_val:.2f} m : ranks={ranks}  horizon={h_str}')

        if h_full is not None and (horizon is None or h_full < horizon):
            horizon = h_full

    if horizon is None:
        horizon = 8
        print('  WARNING: full observability not reached within 8 steps')
    print(f'  Observability horizon = {horizon}  (expected 2)')

    if save_dir is not None:
        _plot_observability(rank_profiles, horizon, save_dir)

    return {'horizon': horizon, 'rank_profiles': rank_profiles}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def recommend_segment_len(fs, fs_new, save_dir, dtype=torch.float64):
    """
    Determine segment_len from step response pole analysis.
    Called by precompute._compute() - prints, no plots.

    Parameters
    ----------
    fs       : native sampling frequency [Hz] (before decimation)
    fs_new   : target sampling frequency [Hz] (after decimation)
    save_dir : directory for cache artefacts (unused here, passed for consistency)
    dtype    : torch dtype (default float64)

    Returns
    -------
    int - segment_len in samples at fs_new
    """
    r_step = _diag_step_response(fs, fs_new, save_dir=None, dtype=dtype)
    return r_step['segment_len']


def _save_report(trajs, r_fft, r_step, r_obs, save_dir):
    """
    Write a compact diagnostics_report.txt — all key numbers, no prose.
    Designed to be pasted directly into a review conversation.
    """
    import datetime
    fs_orig = float(trajs[0]['fs'])
    lines   = []
    W       = 60

    def hdr(title):
        lines.append('=' * W)
        lines.append(f'  {title}')
        lines.append('-' * W)

    def sep():
        lines.append('-' * W)

    lines.append('=' * W)
    lines.append('  DIAGNOSTICS REPORT')
    lines.append(f'  Generated : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'  Trajs     : {", ".join(t["id"] for t in trajs)}')
    lines.append(f'  fs_orig   : {fs_orig:.0f} Hz')
    lines.append('=' * W)

    # ── 1. FFT ──────────────────────────────────────────────────────────
    hdr('1. FFT  (f_99 = freq below which 99% power lies)')
    lines.append(f'  {"Traj":<6}  {"X1 [Hz]":>10}  {"X2 [Hz]":>10}  {"Y [Hz]":>10}')
    lines.append(f'  {"-"*6}  {"-"*10}  {"-"*10}  {"-"*10}')
    for traj, f99s in zip(trajs, r_fft['f99_by_traj']):
        lines.append(f'  {traj["id"]:<6}  {f99s[0]:>10.1f}  {f99s[1]:>10.1f}  {f99s[2]:>10.1f}')
    lines.append(f'  {"MAX":<6}  {"":>10}  {"":>10}  {r_fft["f99_overall"]:>9.1f}*')
    lines.append(f'  * f_99_overall = {r_fft["f99_overall"]:.1f} Hz  '
                 f'(max across all channels + trajs)')
    lines.append(f'  Rule: fs_new >= 8 x {r_fft["f99_overall"]:.0f} = '
                 f'{8 * r_fft["f99_overall"]:.0f} Hz')
    lines.append(f'  -> fs_new = {r_fft["fs_new"]} Hz  '
                 f'(D = {r_fft["decimation_factor"]} from {fs_orig:.0f} Hz)')

    # ── 2. Pole analysis ─────────────────────────────────────────────────
    sep()
    hdr('2. POLE ANALYSIS  (frozen LTI at each Y)')
    lines.append(f'  {"Y [m]":<7}  {"tau_max [s]":>11}  {"rigid-body":>10}  '
                 f'{"unstable":>8}  poles (re+imj)')
    lines.append(f'  {"-"*7}  {"-"*11}  {"-"*10}  {"-"*8}  {"-"*28}')
    for Y_val, eigvals in r_step['poles'].items():
        neg_real   = eigvals.real[eigvals.real < -1e-8]
        tau_Y      = float((-1.0 / neg_real).max()) if neg_real.numel() > 0 else 0.0
        n_rb       = int((eigvals.real.abs() < 1e-4).sum())
        n_unstable = int((eigvals.real > 1e-4).sum())
        pole_str   = '  '.join(
            f'{e.real.item():+.2f}{e.imag.item():+.2f}j' for e in eigvals
        )
        lines.append(f'  {Y_val:<7.2f}  {tau_Y:>11.4f}  {n_rb:>10}  '
                     f'{n_unstable:>8}  {pole_str}')
    lines.append(f'  tau_max_overall = {r_step["tau_max"]:.4f} s')
    lines.append(f'  f_osc_min       = {r_step["f_osc_min"]:.4f} Hz  '
                 f'(slowest oscillatory mode)')
    lines.append(f'  segment_len     = {r_step["segment_len"]} samples  '
                 f'({r_step["segment_len_s"]:.3f} s at {r_fft["fs_new"]:.0f} Hz,  '
                 f'{N_PERIODS} periods)')

    # ── 3. Observability ─────────────────────────────────────────────────
    sep()
    hdr('3. OBSERVABILITY  (rank of O_h vs horizon h)')
    max_h = max(len(v) for v in r_obs['rank_profiles'].values())
    h_cols = ''.join(f'  h={h}' for h in range(1, max_h + 1))
    lines.append(f'  {"Y [m]":<7}{h_cols}  full_at_h')
    lines.append(f'  {"-"*7}' + '  ----' * max_h + '  ---------')
    for Y_val, ranks in r_obs['rank_profiles'].items():
        rank_str  = ''.join(f'  {r:>4}' for r in ranks)
        full_h    = ranks.index(6) + 1 if 6 in ranks else f'>{max_h}'
        lines.append(f'  {Y_val:<7.2f}{rank_str}  {full_h}')
    lines.append(f'  Observability horizon = {r_obs["horizon"]}  (expected 2)')

    # ── Summary ──────────────────────────────────────────────────────────
    sep()
    hdr('SUMMARY')
    lines.append(f'  fs_new      : {r_fft["fs_new"]} Hz  (D={r_fft["decimation_factor"]})')
    lines.append(f'  tau_max     : {r_step["tau_max"]:.4f} s')
    lines.append(f'  f_osc_min   : {r_step["f_osc_min"]:.4f} Hz')
    lines.append(f'  segment_len : {r_step["segment_len"]} samples  '
                 f'({r_step["segment_len_s"]:.3f} s)')
    lines.append(f'  obs_horizon : {r_obs["horizon"]}  (expected 2)')
    lines.append('=' * W)

    path = os.path.join(save_dir, 'diagnostics_report.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Report saved: {path}')


def run_all_diagnostics(trajs, save_dir):
    """
    Run all three diagnostics. Saves plots + diagnostics_report.txt to save_dir.

    Parameters
    ----------
    trajs    : list of traj dicts — each with keys: id, u, q1, state_traj, N, fs
    save_dir : directory to save plots (created if absent)
    """
    os.makedirs(save_dir, exist_ok=True)
    fs = float(trajs[0]['fs'])

    r_fft  = _diag_fft(trajs, save_dir)
    r_step = _diag_step_response(fs, r_fft['fs_new'], save_dir, dtype=torch.float64)
    r_obs  = _diag_observability(fs, save_dir, dtype=torch.float64)

    print()
    print('=' * 60)
    print('  EXPERIMENT DIAGNOSTICS SUMMARY')
    print('-' * 60)
    print(f'  Recommended fs   : {r_fft["fs_new"]} Hz'
          f'  (D={r_fft["decimation_factor"]} from {fs:.0f} Hz)')
    print(f'  tau_max          : {r_step["tau_max"]:.4f} s')
    print(f'  f_osc_min        : {r_step["f_osc_min"]:.4f} Hz')
    print(f'  segment_len      : {r_step["segment_len"]} samples'
          f'  ({r_step["segment_len_s"]:.3f} s at {r_fft["fs_new"]:.0f} Hz)')
    print(f'  Observability    : horizon = {r_obs["horizon"]}  (expected 2)')
    print('=' * 60)

    _save_report(trajs, r_fft, r_step, r_obs, save_dir)


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def _plot_poles(poles, tau_max, save_dir):
    colors = ['steelblue', 'darkorange', 'forestgreen']
    fig, ax = plt.subplots(figsize=(7, 5))
    for (Y_val, eigvals), color in zip(poles.items(), colors):
        re = eigvals.real.numpy()
        im = eigvals.imag.numpy()
        ax.scatter(re, im, c=color, s=60, label=f'Y={Y_val:.2f} m', zorder=3)
    ax.axvline(0, color='k', linewidth=1.0, linestyle='--', label='Stability boundary')
    ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
    ax.set_xlabel('Real part [1/s]')
    ax.set_ylabel('Imaginary part [rad/s]')
    ax.set_title(f'Poles at frozen Y  (tau_max = {tau_max:.3f} s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'diag_step_response.png')


def _plot_observability(rank_profiles, horizon, save_dir):
    colors = ['steelblue', 'darkorange', 'forestgreen']
    fig, ax = plt.subplots(figsize=(6, 4))
    for (Y_val, ranks), color in zip(rank_profiles.items(), colors):
        ax.plot(range(1, len(ranks) + 1), ranks, 'o-', color=color, label=f'Y={Y_val:.2f} m')
    ax.axhline(6, color='k', linestyle='--', linewidth=1.2, label='Full rank (6)')
    ax.set_xlabel('Horizon h [steps]')
    ax.set_ylabel('Rank of O_h')
    ax.set_title(f'Observability rank vs horizon  (full at h={horizon})')
    ax.set_yticks(range(7))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'diag_observability.png')


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
        u, q1, fs_traj = _load_trajectory(mat_path, _dtype)
        state_traj = _build_state_traj_logical(q1, _P_d, _ts_d, _dtype)
        trajs.append({
            'id':         spec['id'],
            'N':          int(q1.shape[0]),
            'fs':         fs_traj,
            'u':          u,
            'q1':         q1,
            'state_traj': state_traj,
        })
        print(f'  {spec["id"]}: T={q1.shape[0]}, fs={fs_traj:.0f} Hz')

    run_all_diagnostics(trajs, save_dir)
