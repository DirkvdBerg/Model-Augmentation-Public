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
    _TRUE_PARAMS, _DETUNED_PARAMS,
)
from lpv_lfr_baseline.core.physics import (
    build_poly_constants, build_M, P as _P, ts as _ts,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate_frozen, simulate

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

_Y_OP_POINTS          = (0.00, 0.20, 0.30)        # frozen Y operating points [m]
_FS_CANDIDATES        = (1000, 2000, 4000, 8000)   # candidate new sampling rates [Hz]
# THEORY: Lecture 9 slides 10-12 (5SMB0) — "10 * omega_b <= omega_s <= 30 * omega_b"
_FS_RULE_FACTOR       = 10                         # require fs_new >= factor * f_osc_min (system bandwidth)
_FFT_ENERGY_THRESHOLD = 0.99                       # cumulative PSD energy for f_99
N_PERIODS             = 3                          # number of slowest oscillatory periods per segment
_N_GRAD_CHECK_EPOCHS  = 50                         # epochs for gradient convergence quick-check
GRAD_CHECK            = False                      # toggle: run diagnostic 4 when running as __main__
_WELCH_FREQ_RESOLUTION_HZ = None                  # None = f_osc_min/3 (auto), or set float [Hz]

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

def _get_f_osc_min(dtype=torch.float64):
    """Return slowest oscillatory mode frequency [Hz] from physics model (no data needed)."""
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


def _diag_fft(trajs, save_dir, *, fs_new, f_osc_min):
    """
    Compute PSD of each output channel for all trajectories.

    Centres signals before FFT to remove the DC operating-point offset.
    Finds f_99: the frequency below which _FFT_ENERGY_THRESHOLD of signal power
    lies per channel. f_99 is reported as a diagnostic check only — it is NOT used
    to set fs_new. fs_new is passed in from run_all_diagnostics (derived from
    f_osc_min via the Lecture 9 "10 * omega_b <= omega_s" rule).

    A warning is printed if f_99 > 10 * f_osc_min (excitation energy above model band).

    Parameters
    ----------
    fs_new     : target sampling frequency [Hz] — already determined from f_osc_min
    f_osc_min  : slowest oscillatory natural frequency [Hz] from pole analysis

    Returns
    -------
    dict: f99_overall [Hz], fs_new [Hz], decimation_factor, f_osc_min [Hz], f99_by_traj
    """
    fs_orig = float(trajs[0]['fs'])

    f99_by_traj = []
    psd_data    = []

    from scipy.signal import welch as _welch
    delta_f = (_get_f_osc_min() / 3.0) if _WELCH_FREQ_RESOLUTION_HZ is None \
              else float(_WELCH_FREQ_RESOLUTION_HZ)
    nperseg = max(256, int(fs_orig / delta_f))
    print(f'  Welch: delta_f={delta_f:.2f} Hz  nperseg={nperseg}'
          f'  (~{int(fs_orig / nperseg)} Hz resolution)')

    with torch.no_grad():
        for traj in trajs:
            q1  = traj['q1']
            q1c = q1 - q1.mean(dim=0, keepdim=True)

            freqs_np, psd_np = _welch(q1c.numpy(), fs=fs_orig, nperseg=nperseg,
                                      window='hann', axis=0)
            freqs = torch.tensor(freqs_np)
            psd   = torch.tensor(psd_np)

            cum  = psd.cumsum(dim=0) / psd.sum(dim=0, keepdim=True).clamp(min=1e-30)
            f99s = []
            for c in range(3):
                hits = (cum[:, c] >= _FFT_ENERGY_THRESHOLD).nonzero(as_tuple=True)[0]
                f99s.append(float(freqs[hits[0]]) if hits.numel() > 0 else fs_orig / 2)
            f99_by_traj.append(f99s)
            psd_data.append((freqs[1:].numpy(), psd[1:].numpy()))

    f99_overall = max(f for f99s in f99_by_traj for f in f99s)
    worst_traj, worst_ch = next(
        (trajs[ti]['id'], _CH_NAMES[ci])
        for ti, f99s in enumerate(f99_by_traj)
        for ci, f in enumerate(f99s)
        if f == f99_overall
    )
    # THEORY: Lecture 9 slides 10-12 (5SMB0) — fs_new is driven by system physics (f_osc_min),
    # not signal content (f_99). D is derived from the passed-in fs_new.
    D = round(fs_orig / fs_new)

    # HEURISTIC: model band cap = 10 * f_osc_min — same factor as the sampling rule;
    # energy above this band aliases after decimation but does not bias in-band estimates
    # (Gonzalez, van Haren, Oomen, Rojas — arXiv:2410.19629 / IEEE TAC 2024)
    model_band = _FS_RULE_FACTOR * f_osc_min

    print('\nFFT Analysis  (f_99 is informational — does NOT drive fs_new)')
    print(f'  fs_original = {fs_orig:.0f} Hz')
    print(f'  model band  = {_FS_RULE_FACTOR} x f_osc_min = {_FS_RULE_FACTOR} x {f_osc_min:.2f} = {model_band:.1f} Hz')
    print(f'  {"traj":<6}' + ''.join(f'  {ch:>10}' for ch in _CH_NAMES))
    for traj, f99s in zip(trajs, f99_by_traj):
        print(f'  {traj["id"]:<6}' + ''.join(f'  {f:>9.1f}Hz' for f in f99s))
    print(f'  f_99 overall (max across all channels + trajectories): {f99_overall:.1f} Hz')
    if f99_overall > model_band:
        print(f'  WARNING: f_99={f99_overall:.0f} Hz > model band ({model_band:.0f} Hz). '
              f'Excitation energy above the model band will alias after decimation '
              f'but should not bias in-band estimates (Gonzalez et al. 2024). '
              f'fs_new is set from physics, not from signal content.')
    print(f'  fs_new = {fs_new} Hz  (set from f_osc_min, see Diagnostic 2; D={D})')

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
            f'FFT  -  f_99={f99_overall:.0f} Hz  (info only)  '
            f'fs_new={fs_new} Hz (from f_osc_min={f_osc_min:.2f} Hz, D={D})',
            fontsize=10,
        )
        plt.tight_layout()
        _save_fig(fig, save_dir, 'diag_fft.png')

    return {'f99_overall': f99_overall, 'fs_new': fs_new, 'decimation_factor': D,
            'f99_by_traj': f99_by_traj, 'worst_traj': worst_traj, 'worst_ch': worst_ch,
            'f_osc_min': f_osc_min}


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

    f_osc_min = min(all_osc_freqs_rads) / (2 * math.pi)
    n_params  = len(_PARAM_NAMES)

    seg_period  = math.ceil(N_PERIODS / f_osc_min * fs_new)        # period rule
    seg_taumax  = math.ceil(10 * tau_max * fs_new)                  # 10x time constant
    seg_nparams = 10 * n_params                                     # 10x parameter count

    # THEORY: Lecture 9 slide 9 (5SMB0) — "N >= 10 * tau_set,95" and "N >= 10 * n_theta"
    # THEORY: Lecture 3 periodic measurement (5SMB0) — integer periods required
    # HEURISTIC: N_PERIODS = 3 — covers slowest mode with margin; Lecture 12 uses 10 for FRF quality
    segment_len   = max(seg_period, seg_taumax, seg_nparams)
    segment_len_s = segment_len / fs_new

    binding_rule = (
        'period rule'        if segment_len == seg_period  and seg_period >= max(seg_taumax, seg_nparams) else
        '10x tau_max rule'   if segment_len == seg_taumax  and seg_taumax >= seg_nparams else
        '10x n_params rule'
    )

    print(f'  f_osc_min = {f_osc_min:.4f} Hz  (slowest oscillatory mode)')
    print(f'  segment_len candidates:')
    print(f'    period rule    ({N_PERIODS}/f_osc_min): {seg_period} samples  ({N_PERIODS/f_osc_min:.3f} s)')
    print(f'    10x tau_max               : {seg_taumax} samples  ({10*tau_max:.3f} s)')
    print(f'    10x n_params  (10x{n_params:d})    : {seg_nparams} samples')
    print(f'  segment_len = {segment_len} samples at {fs_new:.0f} Hz  ({segment_len_s:.3f} s)'
          f'  [binding: {binding_rule}]')

    if seg_taumax > seg_period * 5:
        print(f'  NOTE: 10x tau_max rule dominates strongly ({seg_taumax} vs {seg_period} from period rule).')
        print(f'        This rule is derived for stationary FRF estimation (Lecture 9).')
        print(f'        For BPTT training it may be overly conservative. Discuss with supervisor.')

    if save_dir is not None:
        _plot_poles(poles, tau_max, save_dir)

    return {
        'tau_max':       tau_max,
        'poles':         poles,
        'f_osc_min':     f_osc_min,
        'segment_len':   segment_len,
        'segment_len_s': segment_len_s,
        'binding_rule':  binding_rule,
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
    lines.append(f'  * worst case: {r_fft["worst_traj"]} / {r_fft["worst_ch"]}'
                 f' = {r_fft["f99_overall"]:.1f} Hz  (informational only)')
    model_band = _FS_RULE_FACTOR * r_fft['f_osc_min']
    if r_fft['f99_overall'] > model_band:
        lines.append(f'  WARNING: f_99 ({r_fft["f99_overall"]:.0f} Hz) > model band '
                     f'({model_band:.0f} Hz = {_FS_RULE_FACTOR} x f_osc_min). '
                     f'Out-of-band content aliases but does not bias in-band estimates '
                     f'(Gonzalez et al. 2024).')
    lines.append(f'  fs_new rule: >= {_FS_RULE_FACTOR} x f_osc_min = '
                 f'{_FS_RULE_FACTOR} x {r_fft["f_osc_min"]:.2f} = {model_band:.0f} Hz'
                 f'  [Lecture 9 "10*omega_b <= omega_s"]')
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
                 f'({r_step["segment_len_s"]:.3f} s at {r_fft["fs_new"]:.0f} Hz)'
                 f'  [binding: {r_step["binding_rule"]}]')

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


def run_all_diagnostics(trajs, save_dir, grad_check=False):
    """
    Run all diagnostics. Saves plots + diagnostics_report.txt to save_dir.

    Parameters
    ----------
    trajs      : list of traj dicts — each with keys: id, u, q1, state_traj, N, fs
    save_dir   : directory to save plots (created if absent)
    grad_check : if True, run diagnostic 4 (gradient convergence check)
    """
    os.makedirs(save_dir, exist_ok=True)
    fs = float(trajs[0]['fs'])

    # THEORY: Lecture 9 slides 10-12 (5SMB0) — fs_new derived from system physics (f_osc_min),
    # not from signal content (f_99).  Derivation chain: poles -> f_osc_min -> fs_new.
    f_osc_min_pre = _get_f_osc_min()
    fs_new = next(
        (f for f in _FS_CANDIDATES if f >= _FS_RULE_FACTOR * f_osc_min_pre),
        int(fs),
    )
    print(f'Sampling rate: f_osc_min={f_osc_min_pre:.4f} Hz  ->  '
          f'fs_new={fs_new} Hz  (rule: >= {_FS_RULE_FACTOR} x f_osc_min = '
          f'{_FS_RULE_FACTOR * f_osc_min_pre:.1f} Hz)')

    r_fft  = _diag_fft(trajs, save_dir, fs_new=fs_new, f_osc_min=f_osc_min_pre)
    r_step = _diag_step_response(fs, fs_new, save_dir, dtype=torch.float64)
    r_obs  = _diag_observability(fs, save_dir, dtype=torch.float64)

    print()
    print('=' * 60)
    print('  EXPERIMENT DIAGNOSTICS SUMMARY')
    print('-' * 60)
    print(f'  fs_new           : {r_fft["fs_new"]} Hz'
          f'  (D={r_fft["decimation_factor"]} from {fs:.0f} Hz)'
          f'  [from f_osc_min={r_step["f_osc_min"]:.4f} Hz, Lecture 9]')
    print(f'  tau_max          : {r_step["tau_max"]:.4f} s')
    print(f'  f_osc_min        : {r_step["f_osc_min"]:.4f} Hz')
    print(f'  f_99 overall     : {r_fft["f99_overall"]:.1f} Hz  (informational; not driving fs_new)')
    print(f'  segment_len      : {r_step["segment_len"]} samples'
          f'  ({r_step["segment_len_s"]:.3f} s at {r_fft["fs_new"]:.0f} Hz)'
          f'  [binding: {r_step["binding_rule"]}]')
    print(f'  Observability    : horizon = {r_obs["horizon"]}  (expected 2)')
    print('=' * 60)

    _save_report(trajs, r_fft, r_step, r_obs, save_dir)

    if grad_check:
        _diag_gradient_convergence(trajs, r_fft, r_step, save_dir, dtype=torch.float64)


# ----------------------------------------------------------------------
# Diagnostic 4 — Gradient convergence check
# ----------------------------------------------------------------------

def _diag_gradient_convergence(trajs, r_fft, r_step, save_dir, dtype=torch.float64):
    """
    Gradient convergence sanity check using true parameters.

    Imports LR and W from train_param_recovery so the check uses the same
    optimiser settings as actual training.

    Two checks
    ----------
    1. Gradient direction (1 backward pass at detuned params):
       dot(grad[i], log_true[i]) < 0  →  gradient points toward true value.
    2. Short optimisation (_N_GRAD_CHECK_EPOCHS of Adam):
       tracks per-parameter % error vs true params; checks that errors decrease.

    Returns
    -------
    dict with direction and convergence results, or None if trajectories too short.
    """
    # Lazy import to avoid circular dependency at module load time
    from lpv_lfr_baseline.scripts.train_param_recovery import LR, W as W_cfg
    from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical

    D           = r_fft['decimation_factor']
    fs_new      = float(r_fft['fs_new'])
    ts_eff      = 1.0 / fs_new
    segment_len = r_step['segment_len']
    W_eff       = segment_len if W_cfg is None else int(W_cfg)
    n_windows   = (segment_len + W_eff - 1) // W_eff

    print('\nGradient Convergence Check')
    print(f'  LR={LR}  W={"full" if W_cfg is None else W_cfg}'
          f'  (W_eff={W_eff})  segment_len={segment_len}  n_windows={n_windows}')

    # --- Decimate and build state trajectories ---
    from scipy.signal import decimate as _decimate
    P        = _P.to(dtype)
    dec_trajs = []
    for traj in trajs:
        # THEORY: Lecture 9 (5SMB0) pre-processing — apply anti-aliasing filter before downsampling
        # THEORY: lecture_digital-filters.pdf (4CM00) slides 30-35 — >= 40 dB at new Nyquist
        # scipy.signal.decimate applies Chebyshev Type I filter automatically before striding
        q1_dec = torch.tensor(
            _decimate(traj['q1'].numpy(), D, axis=0), dtype=traj['q1'].dtype
        )
        u_dec  = torch.tensor(
            _decimate(traj['u'].numpy(), D, axis=1), dtype=traj['u'].dtype
        )
        N_dec  = int(q1_dec.shape[0])
        if N_dec < segment_len:
            print(f'  WARNING: {traj["id"]} too short after decimation '
                  f'({N_dec} < {segment_len}) — skipping')
            continue
        state = _build_state_traj_logical(q1_dec, P, ts_eff, dtype)
        dec_trajs.append({'id': traj['id'], 'u': u_dec, 'q1': q1_dec,
                          'state_traj': state, 'N': N_dec})

    if not dec_trajs:
        print('  ERROR: no trajectories long enough — aborting gradient check')
        return None

    # Batch: first segment from each trajectory (same as one training epoch)
    x0_batch  = torch.stack([t['state_traj'][0]          for t in dec_trajs])  # (B, 6)
    u_batch   = torch.stack([t['u'][0, :segment_len]      for t in dec_trajs])  # (B, S, 3)
    q1_batch  = torch.stack([t['q1'][:segment_len]        for t in dec_trajs])  # (B, S, 3)

    # Global sigma from all decimated data (same logic as precompute 'global' mode)
    q1_all = torch.cat([t['q1'] for t in dec_trajs], dim=0)
    sigma  = torch.stack([q1_all[:, c].std().clamp(min=1e-4) for c in range(3)])  # (3,)

    # True log_params: direction from detuned (log_params=0) toward true params
    params_true_t    = torch.tensor([_TRUE_PARAMS[n]    for n in _PARAM_NAMES], dtype=dtype)
    params_detuned_t = torch.tensor([_DETUNED_PARAMS[n] for n in _PARAM_NAMES], dtype=dtype)
    log_params_true  = (params_true_t / params_detuned_t).log()   # target for log_params

    ts_tensor = torch.tensor(ts_eff, dtype=dtype)

    def _loss(block):
        """Windowed forward loss — differentiable w.r.t. block.log_params."""
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_matrices(
            block.log_params, block.params_init, dtype
        )
        x_win    = x0_batch
        total    = None
        for w in range(n_windows):
            ws      = w * W_eff
            we      = min(ws + W_eff, segment_len)
            result  = simulate(x_win, u_batch[:, ws:we], G, K, C, mh, alpha, beta, gamma,
                               N0, N1, N2, block._P, ts_tensor,
                               bptt_mode='full', return_latents=False)
            err     = (result.Y - q1_batch[:, ws:we]) / sigma.unsqueeze(0).unsqueeze(0)
            wloss   = err.pow(2).mean() / n_windows
            total   = wloss if total is None else total + wloss
            x_win   = result.X[:, -1, :].detach()
        return total

    # ── 1. Gradient direction test ────────────────────────────────────────
    block = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=dtype)
    block.log_params.data.zero_()   # detuned starting point

    _loss(block).backward()
    grad = block.log_params.grad.detach().clone()

    # dot < 0  →  gradient descent moves log_params toward log_params_true
    dots    = (grad * log_params_true).tolist()
    correct = [d < 0 for d in dots]
    n_ok    = sum(correct)

    print(f'\n  [1] Gradient direction  (dot < 0 = grad points toward true value)'
          f'  {n_ok}/{len(_PARAM_NAMES)} correct')
    print(f'  {"Param":<8}  {"grad":>12}  {"to_true":>12}  {"dot":>12}  correct?')
    print(f'  {"-"*8}  {"-"*12}  {"-"*12}  {"-"*12}  --------')
    for i, name in enumerate(_PARAM_NAMES):
        print(f'  {name:<8}  {float(grad[i]):>12.4e}  {float(log_params_true[i]):>12.4e}'
              f'  {dots[i]:>12.4e}  {"YES" if correct[i] else "NO"}')

    # ── 2. Short optimisation ─────────────────────────────────────────────
    block = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=dtype)
    block.log_params.data.zero_()
    optimizer = torch.optim.Adam(block.parameters(), lr=LR)

    def _param_err_pct():
        p = block.params_init * block.log_params.detach().exp()
        return ((p - params_true_t) / params_true_t * 100).abs().tolist()

    init_err     = _param_err_pct()
    loss_history = []

    # Compute initial loss before any optimisation step
    with torch.no_grad():
        loss_history.append(float(_loss(block)))

    print(f'\n  [2] Short optimisation  ({_N_GRAD_CHECK_EPOCHS} epochs, LR={LR})')
    print(f'  {"epoch":>6}  {"loss":>12}  {"grad_norm":>12}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*12}')
    for epoch in range(1, _N_GRAD_CHECK_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_val = _loss(block)
        loss_val.backward()
        grad_norm = float(block.log_params.grad.norm().item()
                          if block.log_params.grad is not None else 0.0)
        optimizer.step()
        loss_history.append(float(loss_val.item()))
        if epoch % 10 == 0 or epoch == _N_GRAD_CHECK_EPOCHS:
            print(f'  {epoch:>6}  {loss_history[-1]:>12.4e}  {grad_norm:>12.4e}', flush=True)

    final_err   = _param_err_pct()
    n_improved  = sum(fe < ie for ie, fe in zip(init_err, final_err))

    print(f'  {"Param":<8}  {"init err%":>9}  {"final err%":>10}  improved?')
    print(f'  {"-"*8}  {"-"*9}  {"-"*10}  ---------')
    for i, name in enumerate(_PARAM_NAMES):
        imp = final_err[i] < init_err[i]
        print(f'  {name:<8}  {init_err[i]:>9.2f}  {final_err[i]:>10.2f}  {"YES" if imp else "NO"}')
    print(f'  -> {n_improved}/{len(_PARAM_NAMES)} parameters improved')

    result = {
        'n_correct_direction': n_ok,
        'n_params':            len(_PARAM_NAMES),
        'dots':                dots,
        'correct':             correct,
        'init_err_pct':        init_err,
        'final_err_pct':       final_err,
        'n_improved':          n_improved,
        'loss_init':           loss_history[0],
        'loss_final':          loss_history[-1],
        'W_eff':               W_eff,
        'LR':                  LR,
        'n_epochs':            _N_GRAD_CHECK_EPOCHS,
    }

    if save_dir is not None:
        _append_gradient_report(result, save_dir)

    return result


def _append_gradient_report(r, save_dir):
    """Append gradient convergence section to diagnostics_report.txt."""
    W = 60
    lines = ['', '=' * W, '  4. GRADIENT CONVERGENCE CHECK',
             f'  LR={r["LR"]}  W_eff={r["W_eff"]}  epochs={r["n_epochs"]}',
             '-' * W]

    lines.append(f'  [1] Direction  ({r["n_correct_direction"]}/{r["n_params"]} correct)')
    lines.append(f'  {"Param":<8}  {"dot":>12}  ok?')
    lines.append(f'  {"-"*8}  {"-"*12}  ---')
    for name, dot, ok in zip(_PARAM_NAMES, r['dots'], r['correct']):
        lines.append(f'  {name:<8}  {dot:>12.4e}  {"YES" if ok else "NO"}')

    lines += ['', f'  [2] Convergence  ({r["n_improved"]}/{r["n_params"]} improved)']
    lines.append(f'  {"Param":<8}  {"init%":>8}  {"final%":>8}  ok?')
    lines.append(f'  {"-"*8}  {"-"*8}  {"-"*8}  ---')
    for name, ie, fe in zip(_PARAM_NAMES, r['init_err_pct'], r['final_err_pct']):
        lines.append(f'  {name:<8}  {ie:>8.2f}  {fe:>8.2f}  {"YES" if fe < ie else "NO"}')
    lines.append(f'  Loss: {r["loss_init"]:.4e} -> {r["loss_final"]:.4e}')
    lines.append('=' * W)

    path = os.path.join(save_dir, 'diagnostics_report.txt')
    with open(path, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Gradient report appended: {path}')


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
    parser.add_argument(
        '--grad-check', action='store_true', default=GRAD_CHECK,
        help=f'Run gradient convergence check (diagnostic 4, {_N_GRAD_CHECK_EPOCHS} epochs)',
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

    run_all_diagnostics(trajs, save_dir, grad_check=args.grad_check)
