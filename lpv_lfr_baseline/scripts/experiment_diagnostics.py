"""
experiment_diagnostics.py
-------------------------
Experiment-level diagnostics for the dual-gantry parameter recovery dataset.

Diagnostics (run in order — each feeds the next)
-------------------------------------------------
1. FFT / frequency content   — sampling rate and decimation factor recommendation
2. Step response             — dominant time constant tau_max per Y operating point
3. Parameter sensitivity     — minimum segment length per parameter via finite differences
4. Observability             — horizon sanity check (expected: 2 samples)

Public API
----------
recommend_segment_len(trajs, fs, save_dir, dtype) -> int
    Called by precompute._compute(). Runs diagnostics 2+3 only (no plots).
    Returns segment_len in samples at the given fs.

run_all_diagnostics(trajs, save_dir) -> None
    Runs all four diagnostics, saves plots, prints full summary.
    Called by __main__.

Run standalone:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics --dataset base
"""

import math
import os

import matplotlib
matplotlib.use('Agg')   # non-interactive — safe on servers, always saves to file
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
_ENERGY_THRESHOLD     = 0.95                       # cumulative sensitivity energy to capture
_FS_CANDIDATES        = (1000, 2000, 4000, 8000)   # candidate new sampling rates [Hz]
_FS_RULE_FACTOR       = 8                          # require fs_new >= factor * f_99
_FFT_ENERGY_THRESHOLD = 0.99                       # cumulative PSD energy for f_99
_SENS_T_TEST_MAX      = 2000                       # max decimated samples for sensitivity JVP
_SENS_T_TEST_FACTOR   = 5                          # T_test = factor * tau_max * fs_dec

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
# Shared helper — differentiable matrix build from log_params
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
# Diagnostic 1 — FFT / frequency content
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
            f'FFT  —  f_99={f99_overall:.0f} Hz  ->  Recommended fs={fs_new} Hz (D={D})',
            fontsize=11,
        )
        plt.tight_layout()
        _save_fig(fig, save_dir, 'diag_fft.png')

    return {'f99_overall': f99_overall, 'fs_new': fs_new, 'decimation_factor': D}


# ----------------------------------------------------------------------
# Diagnostic 2 — Step response / pole analysis
# ----------------------------------------------------------------------

def _diag_step_response(fs, save_dir, dtype=torch.float64):
    """
    Compute eigenvalues of A_c = [[0, I], [-M(Y)^-1 K, -M(Y)^-1 C]] at each
    Y in _Y_OP_POINTS using detuned initial parameters.

    Time constants: tau_i = -1 / Re(lambda_i) for Re(lambda_i) < 0.
    tau_max = max over all poles and operating points.

    Returns
    -------
    dict: tau_max [s], poles {Y_val: eigvals tensor}
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
            M_Y = build_M(torch.tensor(Y_val, dtype=dtype))
            A_c = torch.zeros(6, 6, dtype=dtype)
            A_c[:3, 3:] = torch.eye(3, dtype=dtype)
            A_c[3:, :3] = -torch.linalg.solve(M_Y, K)
            A_c[3:, 3:] = -torch.linalg.solve(M_Y, C)
            eigvals = torch.linalg.eigvals(A_c)   # (6,) complex

        poles[Y_val] = eigvals

        neg_real   = eigvals.real[eigvals.real < -1e-8]
        tau_max_Y  = float((-1.0 / neg_real).max()) if neg_real.numel() > 0 else 0.0
        tau_max    = max(tau_max, tau_max_Y)

        # Re≈0: rigid-body modes (no X1/X2 stiffness open-loop — expected)
        # Re>1e-4: genuinely unstable — warn
        n_rb       = int((eigvals.real.abs() < 1e-4).sum())
        n_unstable = int((eigvals.real > 1e-4).sum())
        note  = f'  ({n_rb} rigid-body)' if n_rb else ''
        warn  = f'  WARNING: {n_unstable} unstable pole(s)' if n_unstable else ''
        print(f'  Y={Y_val:.2f} m : tau_max={tau_max_Y:.4f} s'
              f'  poles = [{", ".join(f"{e.real.item():+.2f}{e.imag.item():+.2f}j" for e in eigvals)}]'
              + note + warn)

    print(f'  tau_max overall = {tau_max:.4f} s')

    if save_dir is not None:
        _plot_poles(poles, tau_max, save_dir)

    return {'tau_max': tau_max, 'poles': poles}


# ----------------------------------------------------------------------
# Diagnostic 3 — Parameter sensitivity via JVP
# ----------------------------------------------------------------------

def _diag_param_sensitivity(trajs, fs, tau_max, save_dir, dtype=torch.float64,
                             fs_dec=_FS_CANDIDATES[0]):
    """
    For each log-parameter, compute the time-resolved sensitivity
        s_i(t) = ||d y(t) / d log_theta_i||_2
    via central finite differences through simulate_frozen.

    Uses decimated inputs (fs_dec Hz) so T_test is manageable.
    Averages over all trajectories; Y is frozen at each traj's initial Y.

    Returns
    -------
    dict: segment_len [samples at native fs], t95 {name: float [s]},
          slowest_param str, segment_len_s float
    """
    block       = ParameterizedLFRBlock(RMSE_baseline=1.0)
    params_init = block.params_init.to(dtype)
    log_p       = block.log_params.detach().to(dtype)   # zeros at init
    P           = _P.to(dtype)

    D      = max(1, round(fs / fs_dec))
    fs_dec = fs / D   # exact decimated rate
    ts_eff = torch.tensor(float(_ts) * D, dtype=dtype)

    # T_test: enough to observe _SENS_T_TEST_FACTOR * tau_max at decimated rate,
    # capped to avoid excessive compute.
    min_T_dec = min(traj['N'] // D for traj in trajs)
    T_test = min(
        int(_SENS_T_TEST_FACTOR * tau_max * fs_dec),
        _SENS_T_TEST_MAX,
        min_T_dec,
    )
    T_test = max(T_test, 50)   # floor: always simulate at least 50 samples

    n_params = len(_PARAM_NAMES)
    n_trajs  = len(trajs)

    print(f'\nParameter Sensitivity'
          f'  (fs_dec={fs_dec:.0f} Hz, D={D}, T_test={T_test} steps = {T_test/fs_dec:.2f} s)')

    sensitivity_sum = torch.zeros(n_params, T_test, dtype=dtype)

    for traj in trajs:
        u_dec = traj['u'][0, ::D, :][:T_test].to(dtype)   # (T_test, 3)
        u_seq = u_dec.unsqueeze(0)                          # (1, T_test, 3)
        x0    = traj['state_traj'][0:1].to(dtype)           # (1, 6)
        Y_freeze = float(x0[0, 2].item())                   # Y position at traj start

        def forward_fn(lp):
            with torch.no_grad():
                G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_matrices(
                    lp, params_init, dtype
                )
                result = simulate_frozen(
                    x0, u_seq, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
                    P, ts_eff, Y_freeze=Y_freeze, return_latents=False,
                )
            return result.Y[0]   # (T_test, 3)

        eps = 1e-5
        for i in range(n_params):
            lp_plus  = log_p.clone(); lp_plus[i]  = lp_plus[i]  + eps
            lp_minus = log_p.clone(); lp_minus[i] = lp_minus[i] - eps
            sens_i = (forward_fn(lp_plus) - forward_fn(lp_minus)) / (2 * eps)
            sensitivity_sum[i] += sens_i.norm(dim=-1)
            print(f'    {traj["id"]}  param {_PARAM_NAMES[i]:<6} done', flush=True)

    sensitivity_avg = sensitivity_sum / n_trajs   # (n_params, T_test)

    # Cumulative energy and t_95 per parameter
    t95 = {}
    for i, name in enumerate(_PARAM_NAMES):
        s     = sensitivity_avg[i]
        total = s.pow(2).sum().clamp(min=1e-30)
        cs    = s.pow(2).cumsum(0) / total
        hits  = (cs >= _ENERGY_THRESHOLD).nonzero(as_tuple=True)[0]
        t95_samp    = int(hits[0]) if hits.numel() > 0 else T_test - 1
        t95[name]   = t95_samp / fs_dec

    slowest      = max(t95, key=t95.get)
    t_sens_max   = t95[slowest]
    t_capped     = t_sens_max >= (T_test - 1) / fs_dec * 0.99

    segment_len_s = t_sens_max
    segment_len   = math.ceil(segment_len_s * fs)

    print(f'\n  t_95 per parameter [s]:')
    for name in _PARAM_NAMES:
        cat  = _PARAM_CATEGORIES[name]
        mark = ' <-- slowest' if name == slowest else ''
        print(f'    {name:<6}: {t95[name]:.4f} s  ({cat}){mark}')
    if t_capped:
        print(f'  WARNING: {slowest} sensitivity may not have converged — T_test cap reached')
    print(f'  Segment length: t_95_max = {segment_len_s:.4f} s')
    print(f'  -> segment_len = {segment_len} samples at {fs:.0f} Hz')

    if save_dir is not None:
        _plot_sensitivity(sensitivity_avg, t95, fs_dec, T_test, save_dir)

    return {
        'segment_len':   segment_len,
        'segment_len_s': segment_len_s,
        't95':           t95,
        'slowest_param': slowest,
    }


# ----------------------------------------------------------------------
# Diagnostic 4 — Observability
# ----------------------------------------------------------------------

def _diag_observability(fs, save_dir, dtype=torch.float64):
    """
    Build observability matrix O_h = [C; CA_d; ...; CA_d^{h-1}] at each
    Y in _Y_OP_POINTS and track rank growth.

    C = [P^T | 0_{3x3}]  (3x6): output is y_stage = q_logical @ P, so
    in column-vector convention C selects first 3 states and applies P^T.

    Expected: rank reaches 6 at h=2 since
        [C; CA_c] = [[P^T, 0], [0, P^T]] — block diagonal of invertible P^T.

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
            M_Y = build_M(torch.tensor(Y_val, dtype=dtype))
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

def recommend_segment_len(trajs, fs, save_dir, dtype=torch.float64):
    """
    Determine segment_len from step response + parameter sensitivity.
    Called by precompute._compute() — prints, no plots.

    Parameters
    ----------
    trajs    : list of traj dicts (id, u, q1, state_traj, N, fs)
    fs       : native sampling frequency [Hz]
    save_dir : directory for cache artefacts (unused here, passed for consistency)
    dtype    : torch dtype (default float64)

    Returns
    -------
    int — segment_len in samples at the native fs
    """
    print('\nSegment length recommendation')
    r_step = _diag_step_response(fs, save_dir=None, dtype=dtype)
    r_sens = _diag_param_sensitivity(
        trajs, fs, r_step['tau_max'], save_dir=None, dtype=dtype,
        fs_dec=_FS_CANDIDATES[0],
    )
    return r_sens['segment_len']


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

    r_fft  = _diag_fft(trajs, save_dir)
    r_step = _diag_step_response(fs, save_dir, dtype=torch.float64)
    r_sens = _diag_param_sensitivity(
        trajs, fs, r_step['tau_max'], save_dir, dtype=torch.float64,
        fs_dec=r_fft['fs_new'],
    )
    r_obs  = _diag_observability(fs, save_dir, dtype=torch.float64)

    print()
    print('=' * 60)
    print('  EXPERIMENT DIAGNOSTICS SUMMARY')
    print('-' * 60)
    print(f'  Recommended fs   : {r_fft["fs_new"]} Hz'
          f'  (D={r_fft["decimation_factor"]} from {fs:.0f} Hz)')
    print(f'  tau_max          : {r_step["tau_max"]:.4f} s')
    print(f'  Slowest param    : {r_sens["slowest_param"]}'
          f'  (t_95={r_sens["t95"][r_sens["slowest_param"]]:.4f} s)')
    print(f'  segment_len      : {r_sens["segment_len"]} samples'
          f'  ({r_sens["segment_len_s"]:.3f} s at {fs:.0f} Hz)')
    print(f'  Observability    : horizon = {r_obs["horizon"]}  (expected 2)')
    print('=' * 60)


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


def _plot_sensitivity(sensitivity_avg, t95, fs_dec, T_test, save_dir):
    t_axis = [k / fs_dec for k in range(T_test)]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(_PARAM_NAMES):
        cat   = _PARAM_CATEGORIES[name]
        color = _CATEGORY_COLORS[cat]
        s     = sensitivity_avg[i]
        total = s.pow(2).sum().clamp(min=1e-30)
        cs    = (s.pow(2).cumsum(0) / total).tolist()
        ax.plot(t_axis, cs, color=color, alpha=0.85, linewidth=1.2, label=name)
        ax.axvline(t95[name], color=color, linestyle=':', linewidth=0.8, alpha=0.6)
    ax.axhline(_ENERGY_THRESHOLD, color='k', linestyle='--', linewidth=1.2,
               label=f'{_ENERGY_THRESHOLD:.0%} threshold')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Cumulative sensitivity energy [-]')
    ax.set_title('Parameter sensitivity — cumulative energy  (dotted lines = t_95 per param)')
    ax.legend(fontsize=7, ncol=5, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    _save_fig(fig, save_dir, 'diag_sensitivity.png')


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
        '--quick', action='store_true',
        help='Cap sensitivity T_test to 50 samples — fast smoke test, results not meaningful.',
    )
    args = parser.parse_args()

    if args.quick:
        import lpv_lfr_baseline.scripts.experiment_diagnostics as _self
        _self._SENS_T_TEST_MAX = 50
        print('[quick mode] T_test capped to 50 samples — smoke test only')

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
