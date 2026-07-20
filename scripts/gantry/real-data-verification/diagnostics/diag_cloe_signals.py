"""
diag_cloe_signals.py
---------------------
Verify the signal decomposition required for CLOE (Closed-Loop Output Error).

Signal hypothesis (from MATLAB runFDILCAllHostSwLog.m old comments):
    MF30  = total current command   [A]  (feedforward + feedback)
    MF230 = feedback controller output [A]
    MF30 - MF230 = ILC feedforward [A]

Four checks:
    1. iter0 sanity: feedforward OFF -> MF30 ~= MF230 if MF230 = feedback
    2. Feedforward evolution: (MF30 - MF230) grows with ILC iteration number
    3. Filter reconstruction: K(M2) ~= MF230 using Filter1->Filter2 from Telica.mat
       with correct unit chain: M2[um] * 1024 counts/um -> F1 -> F2 -> * AmplifierGain [A]
    4. iter0 CLOE consistency: in iter0 feedforward=0 so MF30 = K(M2) exactly;
       use this as clean ground truth to confirm the unit chain is correct.

Run as:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_cloe_signals.py
"""

__project_origin__ = "added"

import os
import sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))

# -- Config -------------------------------------------------------------------
OP_FOLDER   = 'xpos_-60_ypos-40'
_DATA_ROOT  = os.path.join(
    _ROOT, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL', 'train', OP_FOLDER
)
_SAVE_DIR   = os.path.join(
    _ROOT, 'simulations', 'gantry_subnet', 'diagnostics', 'cloe_signals'
)
_TELICA_MAT = os.path.join(_ROOT, 'kamtin-data', 'Telica.mat')

# Iterations used in each check (skips files that do not exist)
_ITER_SPECS = [
    ('iter0',    'iter0.log'),
    ('iterETEL', 'iterETEL.log'),
    ('iter5',    'iter5.log'),
    ('iter8',    'iter8.log'),
]

_BH       = 'BHL'
_AX_CODES = ('GTRX1', 'GTRX2', 'GTRY')
_AX_NAMES = ('X1',    'X2',    'Y')

_FS_NATIVE = 10_000.0   # Hz -- native log rate
_FS_TARGET = 20_000.0   # Hz -- target after upsampling


# -- Loader -------------------------------------------------------------------

def _load_log_raw(path):
    """
    Load M0, M2, MF30, MF230 for BHL from one iter*.log.
    Positions in um (raw counts), currents in A.
    Upsampled to 20 kHz. Trimmed to 50 ms before first setpoint motion.
    Returns dict: M0 (T,3), M2 (T,3), MF30 (T,3), MF230 (T,3), T, fs.
    """
    with open(path, 'r') as fh:
        header = fh.readline()
    cols = [f.strip().split(':')[0].replace('.', '_')
            for f in header.strip().split('\t')]
    while cols and cols[-1] == '':
        cols.pop()

    raw = pd.read_csv(path, sep='\t', header=None, names=cols,
                      skiprows=1, engine='python', index_col=False)
    raw = raw.dropna(axis=1, how='all')

    M0    = raw[[f'{_BH}_{ax}_M0'    for ax in _AX_CODES]].to_numpy(float)
    M2    = raw[[f'{_BH}_{ax}_M2'    for ax in _AX_CODES]].to_numpy(float)
    MF30  = raw[[f'{_BH}_{ax}_MF30'  for ax in _AX_CODES]].to_numpy(float)
    MF230 = raw[[f'{_BH}_{ax}_MF230' for ax in _AX_CODES]].to_numpy(float)

    # Trim: keep 50 ms before first setpoint deviation
    deviate = np.any(np.abs(M0 - M0[0]) > 0.5, axis=1)   # 0.5 um threshold
    idx     = int(np.argmax(deviate)) if deviate.any() else 0
    trim    = max(0, idx - int(round(50e-3 * _FS_NATIVE)))
    M0, M2, MF30, MF230 = M0[trim:], M2[trim:], MF30[trim:], MF230[trim:]

    # Upsample 10 kHz -> 20 kHz (linear interpolation, matches MATLAB pipeline)
    t_orig = np.arange(len(M0)) / _FS_NATIVE
    t_new  = np.arange(0, t_orig[-1] + 1 / _FS_TARGET, 1 / _FS_TARGET)
    t_new  = t_new[t_new <= t_orig[-1]]

    def _up(sig):
        return interp1d(t_orig, sig, axis=0, kind='linear',
                        bounds_error=False,
                        fill_value=(sig[0], sig[-1]))(t_new)

    return {
        'M0':    _up(M0),     # (T, 3)  um  setpoint
        'M2':    _up(M2),     # (T, 3)  um  tracking error (per schema: M0 - position)
        'MF30':  _up(MF30),   # (T, 3)  A   total current
        'MF230': _up(MF230),  # (T, 3)  A   feedback? (hypothesis being tested)
        'T':     len(t_new),
        'fs':    _FS_TARGET,
    }


# -- Filter loader ------------------------------------------------------------

def _load_filters():
    """
    Extract Filter1, Filter2, AmplifierGain, and EncoderResolution from Telica.mat.

    Unit chain for CLOE feedback reconstruction:
        e [um]  ->  * (1e-6 / enc_res)  ->  e [counts]
        e [counts]  ->  Filter1 -> Filter2  ->  ci [current-increments]
        ci  ->  * amp_gain  ->  u_fb [A]

    Returns dict with keys:
        'X':        [(b1,a1), (b2,a2)] for X axis filters
        'Y':        [(b1,a1), (b2,a2)] for Y axis filters
        'amp_gain': float  A/ci  (same for X and Y, from X axis)
        'enc_res':  float  m/count  (same for X and Y, from X axis)
    """
    mat  = loadmat(_TELICA_MAT, squeeze_me=True, struct_as_record=False)
    axes = mat['MachineParam'].Axes

    def _coeff(filt_obj):
        vals   = filt_obj.Values
        # Telica.mat stores b0-b6 and a1-a6 (7 coefficients each, some trailing zeros)
        b      = np.array([float(getattr(vals, f'b{i}')) for i in range(7)])
        a_rest = np.array([float(getattr(vals, f'a{i}')) for i in range(1, 7)])
        # Trim trailing zeros so lfilter gets the minimal-order filter
        b      = np.trim_zeros(b, 'b') if np.any(b != 0) else np.array([1.0])
        a_rest = np.trim_zeros(a_rest, 'b')
        a      = np.concatenate([[1.0], a_rest])
        return b, a

    amp_gain = float(axes.X.ElectronicHardwareInfo.Motor.AmplifierGain.Value)
    enc_res  = float(axes.X.ElectronicHardwareInfo.EncoderResolution.Value)

    return {
        'X':        [_coeff(axes.X.Controllers.Filter1),
                     _coeff(axes.X.Controllers.Filter2)],
        'Y':        [_coeff(axes.Y.Controllers.Filter1),
                     _coeff(axes.Y.Controllers.Filter2)],
        'amp_gain': amp_gain,   # 0.0020751953125 A/ci
        'enc_res':  enc_res,    # 9.765625e-10 m/count  =>  1 um = 1024 counts
    }


# -- Helpers ------------------------------------------------------------------

def _nrms(pred, meas):
    """NRMS = RMSE / std(meas) * 100 [%]."""
    sigma = np.std(meas)
    if sigma < 1e-12:
        return np.nan
    return float(np.sqrt(np.mean((pred - meas) ** 2)) / sigma * 100.0)


def _corr(a, b):
    """Pearson correlation coefficient."""
    ac = a - a.mean()
    bc = b - b.mean()
    denom = np.sqrt(np.sum(ac ** 2) * np.sum(bc ** 2))
    return float(np.dot(ac, bc) / denom) if denom > 1e-12 else 0.0


def _rms_ratio(pred, meas):
    """RMS(pred) / RMS(meas) -- should be ~1.0 if gain is correct."""
    rms_m = float(np.sqrt(np.mean(meas ** 2)))
    return float(np.sqrt(np.mean(pred ** 2))) / rms_m if rms_m > 1e-12 else np.nan


def _peak_lag_ms(pred, meas, fs, search_ms=10.0):
    """Lag of pred relative to meas [ms], from cross-correlation peak."""
    n = int(search_ms * fs / 1000) * 4
    n = min(n, len(pred))
    pc = pred[:n] - pred[:n].mean()
    mc = meas[:n] - meas[:n].mean()
    xcorr = np.correlate(pc, mc, mode='full')
    lag = int(np.argmax(xcorr)) - (n - 1)
    return lag / fs * 1000.0


def _motion_start(data):
    """Sample index where setpoint first moves (after pre-motion trim)."""
    deviate = np.any(np.abs(data['M0'] - data['M0'][0]) > 0.5, axis=1)
    return int(np.argmax(deviate)) if deviate.any() else 0


def _fkey(aname):
    return 'Y' if aname == 'Y' else 'X'


def _verdict_ff(ratio_pct):
    if ratio_pct < 5:
        return 'PASS  -- feedforward ~0  -> MF230 = feedback'
    if ratio_pct < 20:
        return 'WARN  -- small feedforward (5-20%)'
    return 'FAIL  -- large feedforward (>20%)'


def _verdict_recon(nrms_mot, corr):
    if np.isnan(nrms_mot):
        return 'N/A'
    if nrms_mot < 10 and corr > 0.95:
        return 'CONFIRMED  (NRMS<10%, corr>0.95)'
    if nrms_mot < 30 and corr > 0.80:
        return 'PLAUSIBLE  -- check gain/phase'
    if corr > 0.80:
        return 'GAIN ERROR  -- shape ok, scale wrong'
    return 'POOR  -- check unit chain or filter'


# -- Check 1: iter0 sanity ----------------------------------------------------

def check1_iter0_sanity(data_dict):
    """
    With feedforward OFF (iter0), MF30 ~= MF230 if MF230 is the feedback output.
    Plots MF30 and MF230 side-by-side, and the difference (MF30-MF230).
    """
    if 'iter0' not in data_dict:
        print('\n[Check 1] iter0.log not found -- skipping.')
        return

    d   = data_dict['iter0']
    T   = d['T']
    t_s = np.arange(T) / d['fs']

    print('\n' + '=' * 62)
    print('Check 1: iter0 sanity -- feedforward should be OFF')
    print('  Hypothesis: MF30 ~= MF230  (if MF230 = feedback output)')
    print('=' * 62)
    print(f'  {"Axis":<6}  {"RMS(MF30-MF230)/RMS(MF30)":>28}  Verdict')
    print(f'  {"-"*6}  {"-"*28}  {"-"*50}')

    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex=True)

    for i, aname in enumerate(_AX_NAMES):
        mf30  = d['MF30'][:, i]
        mf230 = d['MF230'][:, i]
        diff  = mf30 - mf230

        rms_total = float(np.sqrt(np.mean(mf30 ** 2))) + 1e-12
        ratio_pct = float(np.sqrt(np.mean(diff ** 2))) / rms_total * 100.0
        print(f'  {aname:<6}  {ratio_pct:>26.1f}%  {_verdict_ff(ratio_pct)}')

        # Left panel: MF30 vs MF230
        ax = axs[i, 0]
        ax.plot(t_s, mf30,  label='MF30  (total)', color='tab:blue',   linewidth=0.7)
        ax.plot(t_s, mf230, label='MF230',          color='tab:orange', linewidth=0.7,
                linestyle='--')
        ax.set_ylabel(f'{aname} [A]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{aname}: MF30 vs MF230 (iter0)', fontsize=9)

        # Right panel: difference (should be ~0 for iter0)
        ax2 = axs[i, 1]
        ax2.plot(t_s, diff, color='tab:red', linewidth=0.7, label='MF30 - MF230')
        ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax2.set_ylabel(f'{aname} [A]')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'{aname}: feedforward? (ratio = {ratio_pct:.1f}%)', fontsize=9)

    axs[-1, 0].set_xlabel('Time [s]')
    axs[-1, 1].set_xlabel('Time [s]')
    fig.suptitle(
        'Check 1: iter0 (feedforward OFF)\n'
        'Left: MF30 vs MF230  |  Right: MF30-MF230  (should be ~0 if MF230 = feedback)',
        fontsize=10
    )
    fig.tight_layout()
    path = os.path.join(_SAVE_DIR, 'check1_iter0_sanity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {path}')
    plt.close(fig)


# -- Check 2: feedforward evolution -------------------------------------------

def check2_ff_evolution(data_dict):
    """
    MF30 - MF230 should grow in amplitude as ILC iterations progress.
    Plots the presumed feedforward signal per axis for each available iteration.
    """
    available = [(n, data_dict[n]) for n, _ in _ITER_SPECS if n in data_dict]
    if len(available) < 2:
        print('\n[Check 2] Need at least 2 iterations -- skipping.')
        return

    print('\n' + '=' * 62)
    print('Check 2: ILC feedforward evolution (MF30 - MF230)')
    print('  RMS should grow with iteration number')
    print('=' * 62)
    print(f'  {"Iteration":<12}  {"X1 RMS [A]":>12}  {"X2 RMS [A]":>12}  {"Y RMS [A]":>12}')
    print(f'  {"-"*12}  {"-"*12}  {"-"*12}  {"-"*12}')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    fig, axs = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    T_min = min(d['T'] for _, d in available)
    t_s   = np.arange(T_min) / available[0][1]['fs']

    for j, (name, d) in enumerate(available):
        ff  = d['MF30'] - d['MF230']              # (T, 3)
        rms = np.sqrt(np.mean(ff ** 2, axis=0))   # (3,)
        print(f'  {name:<12}  {rms[0]:>12.5f}  {rms[1]:>12.5f}  {rms[2]:>12.5f}')

        col = colors[j % len(colors)]
        for i, aname in enumerate(_AX_NAMES):
            axs[i].plot(t_s, ff[:T_min, i], label=name, linewidth=0.7, color=col)

    for i, aname in enumerate(_AX_NAMES):
        axs[i].axhline(0, color='k', linewidth=0.5, linestyle='--')
        axs[i].set_ylabel(f'{aname} [A]')
        axs[i].legend(fontsize=8, loc='upper right')
        axs[i].grid(True, alpha=0.3)
        axs[i].set_title(f'{aname}: MF30 - MF230 per iteration', fontsize=9)

    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(
        'Check 2: ILC feedforward evolution (MF30 - MF230)\n'
        'Amplitude should increase with iteration number if MF30-MF230 = feedforward',
        fontsize=10
    )
    fig.tight_layout()
    path = os.path.join(_SAVE_DIR, 'check2_ff_evolution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {path}')
    plt.close(fig)


# -- Check 3: filter reconstruction (corrected unit chain) --------------------

def _reconstruct_feedback(data, filters, axis_idx, aname):
    """
    Apply the full unit chain for one axis:
        M2 [um]  ->  * enc_per_um  ->  [counts]
        [counts]  ->  Filter1 -> Filter2  ->  [ci]
        [ci]  ->  * amp_gain  ->  [A]

    Filter runs from zero initial state (filter warm-up visible in first ~50ms).
    Returns reconstructed u_fb [A] as ndarray (T,).
    """
    fk             = _fkey(aname)
    (b1, a1), (b2, a2) = filters[fk]
    enc_per_um     = 1e-6 / filters['enc_res']   # counts/um = 1024
    amp_gain       = filters['amp_gain']           # A/ci

    e_counts       = data['M2'][:, axis_idx] * enc_per_um
    ci             = lfilter(b2, a2, lfilter(b1, a1, e_counts))
    return ci * amp_gain


def check3_filter_reconstruction(data_dict, filters):
    """
    Reconstruct MF230 from M2 using the Filter1->Filter2 cascade and correct units.

    Unit chain (from Telica.mat):
        M2 [um] * (1e-6 m/um) / (9.766e-10 m/count) = M2 * 1024  [counts]
        Filter1(Filter2(e_counts))  [current-increments, ci]
        ci * AmplifierGain (0.002075 A/ci)  [A]

    Metrics per axis (motion portion = after pre-motion trim settles):
        NRMS [%]      -- shape + scale error combined
        Correlation   -- shape match independent of scale
        RMS ratio     -- gain check: should be ~1.0
        Peak lag [ms] -- phase check: should be ~0 ms

    Three panels per axis: full trajectory | 200 ms zoom | residual.
    """
    ref_name = next(
        (n for n in ['iterETEL', 'iter5', 'iter8'] if n in data_dict), None
    )
    if ref_name is None:
        print('\n[Check 3] No non-iter0 log available -- skipping.')
        return

    d          = data_dict[ref_name]
    T          = d['T']
    fs         = d['fs']
    t_s        = np.arange(T) / fs
    mot_idx    = _motion_start(d)
    t_mot      = t_s[mot_idx]
    zoom_end   = min(mot_idx + int(0.20 * fs), T)
    t_zoom     = t_s[mot_idx:zoom_end]

    enc_per_um = 1e-6 / filters['enc_res']
    amp_gain   = filters['amp_gain']

    print('\n' + '=' * 62)
    print(f'Check 3: Filter reconstruction K(M2) vs MF230  [{ref_name}]')
    print(f'  Unit chain: M2[um] * {enc_per_um:.0f} -> F1 -> F2 -> * {amp_gain:.7f} [A]')
    print(f'  Motion starts at sample {mot_idx} ({t_mot*1000:.1f} ms)')
    print(f'  Metrics computed on motion portion (sample {mot_idx} onward)')
    print('=' * 62)
    print(f'\n  {"Axis":<5}  {"NRMS% full":>12}  {"NRMS% motion":>14}  '
          f'{"Corr":>8}  {"RMS ratio":>10}  {"Lag [ms]":>10}  Verdict')
    print(f'  {"-"*5}  {"-"*12}  {"-"*14}  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*34}')

    recs = {}
    for i, aname in enumerate(_AX_NAMES):
        u_fb  = _reconstruct_feedback(d, filters, i, aname)
        mf230 = d['MF230'][:, i]
        recs[aname] = u_fb

        nrms_full = _nrms(u_fb, mf230)

        if mot_idx < T - 10:
            u_m   = u_fb[mot_idx:]
            mf_m  = mf230[mot_idx:]
            nrms_mot  = _nrms(u_m, mf_m)
            corr      = _corr(u_m, mf_m)
            ratio     = _rms_ratio(u_m, mf_m)
            lag_ms    = _peak_lag_ms(u_m, mf_m, fs)
        else:
            nrms_mot = corr = ratio = lag_ms = np.nan

        verdict = _verdict_recon(nrms_mot, corr)
        print(f'  {aname:<5}  {nrms_full:>12.1f}  {nrms_mot:>14.1f}  '
              f'{corr:>8.3f}  {ratio:>10.3f}  {lag_ms:>10.2f}  {verdict}')

    print()
    print('  Interpretation guide:')
    print('    NRMS% motion < 10 + corr > 0.95  => unit chain confirmed correct')
    print('    corr > 0.80 but NRMS large        => gain/scale error (wrong amp_gain or enc_res)')
    print('    corr < 0.80                        => wrong input signal or filter structure')
    print('    RMS ratio ~1.0                     => amplitude scale correct')
    print('    Lag ~0 ms                          => no phase shift (filter initial state ok)')

    # -- Plot: 3 rows x 3 cols (full | zoom | residual) -----------------------
    fig, axs = plt.subplots(3, 3, figsize=(16, 9))

    for i, aname in enumerate(_AX_NAMES):
        mf230 = d['MF230'][:, i]
        u_fb  = recs[aname]
        resid = mf230 - u_fb

        kw_mf  = dict(color='tab:blue',   linewidth=0.8, label='MF230 (target)')
        kw_rec = dict(color='tab:orange',  linewidth=0.8, label='K(M2) reconstructed',
                      linestyle='--')
        kw_res = dict(color='tab:red',     linewidth=0.7, label='residual MF230 - K(M2)')

        # Col 0: full trajectory
        ax = axs[i, 0]
        ax.plot(t_s, mf230, **kw_mf)
        ax.plot(t_s, u_fb,  **kw_rec)
        ax.axvline(t_mot, color='gray', linewidth=0.8, linestyle=':', label='motion start')
        ax.set_ylabel(f'{aname} [A]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Full trajectory', fontsize=9)

        # Col 1: zoom -- first 200 ms of motion
        ax2 = axs[i, 1]
        ax2.plot(t_zoom, mf230[mot_idx:zoom_end], **kw_mf)
        ax2.plot(t_zoom, u_fb[mot_idx:zoom_end],  **kw_rec)
        ax2.set_ylabel(f'{aname} [A]')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.set_title('Zoom: first 200 ms of motion\n(filter warm-up visible here)', fontsize=9)

        # Col 2: residual
        ax3 = axs[i, 2]
        ax3.plot(t_s, resid, **kw_res)
        ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax3.axvline(t_mot, color='gray', linewidth=0.8, linestyle=':')
        ax3.set_ylabel(f'{aname} [A]')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        if i == 0:
            ax3.set_title('Residual (MF230 - K(M2))', fontsize=9)

    for col in range(3):
        axs[-1, col].set_xlabel('Time [s]')

    fig.suptitle(
        f'Check 3: Filter reconstruction K(M2) vs MF230  [{ref_name}]\n'
        f'Chain: M2[um] * {enc_per_um:.0f} -> F1 -> F2 -> * {amp_gain:.5f} -> [A]\n'
        f'Vertical dotted line = motion start',
        fontsize=9
    )
    fig.tight_layout()
    path = os.path.join(_SAVE_DIR, 'check3_filter_reconstruction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {path}')
    plt.close(fig)


# -- Check 4: iter0 CLOE consistency ------------------------------------------

def check4_iter0_cloe_consistency(data_dict, filters):
    """
    In iter0 feedforward is OFF, so MF30 = MF230 = K(M2) exactly.
    This is the cleanest ground truth: compare K(M2) reconstruction directly
    against MF30 (no feedforward ambiguity).

    If NRMS < 10% and corr > 0.95 here, the full unit chain is confirmed correct
    and CLOE can be implemented with confidence.
    """
    if 'iter0' not in data_dict:
        print('\n[Check 4] iter0.log not found -- skipping.')
        return

    d       = data_dict['iter0']
    T       = d['T']
    fs      = d['fs']
    t_s     = np.arange(T) / fs
    mot_idx = _motion_start(d)
    t_mot   = t_s[mot_idx]
    zoom_end = min(mot_idx + int(0.20 * fs), T)
    t_zoom  = t_s[mot_idx:zoom_end]

    enc_per_um = 1e-6 / filters['enc_res']
    amp_gain   = filters['amp_gain']

    print('\n' + '=' * 62)
    print('Check 4: iter0 CLOE consistency')
    print('  In iter0: feedforward = 0  =>  MF30 = K(M2) exactly')
    print('  Target signal: MF30  (not MF230, though they are equal in iter0)')
    print('  A good match here CONFIRMS the unit chain for CLOE.')
    print('=' * 62)
    print(f'\n  {"Axis":<5}  {"NRMS% full":>12}  {"NRMS% motion":>14}  '
          f'{"Corr":>8}  {"RMS ratio":>10}  {"Lag [ms]":>10}  Verdict')
    print(f'  {"-"*5}  {"-"*12}  {"-"*14}  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*34}')

    recs = {}
    for i, aname in enumerate(_AX_NAMES):
        u_fb = _reconstruct_feedback(d, filters, i, aname)
        mf30 = d['MF30'][:, i]   # = MF230 in iter0
        recs[aname] = u_fb

        nrms_full = _nrms(u_fb, mf30)

        if mot_idx < T - 10:
            u_m   = u_fb[mot_idx:]
            mf_m  = mf30[mot_idx:]
            nrms_mot = _nrms(u_m, mf_m)
            corr     = _corr(u_m, mf_m)
            ratio    = _rms_ratio(u_m, mf_m)
            lag_ms   = _peak_lag_ms(u_m, mf_m, fs)
        else:
            nrms_mot = corr = ratio = lag_ms = np.nan

        verdict = _verdict_recon(nrms_mot, corr)
        print(f'  {aname:<5}  {nrms_full:>12.1f}  {nrms_mot:>14.1f}  '
              f'{corr:>8.3f}  {ratio:>10.3f}  {lag_ms:>10.2f}  {verdict}')

    print()
    print('  If CONFIRMED here: CLOE implementation is unblocked.')
    print('  If GAIN ERROR:     check AmplifierGain value in Telica.mat.')
    print('  If POOR:           check encoder resolution or filter order.')

    # -- Plot: 3 rows x 3 cols (full | zoom | residual) -----------------------
    fig, axs = plt.subplots(3, 3, figsize=(16, 9))

    for i, aname in enumerate(_AX_NAMES):
        mf30 = d['MF30'][:, i]
        u_fb = recs[aname]
        resid = mf30 - u_fb

        kw_mf  = dict(color='tab:blue',   linewidth=0.8,
                      label='MF30 (= K(M2) ground truth in iter0)')
        kw_rec = dict(color='tab:orange',  linewidth=0.8,
                      label='K(M2) reconstructed', linestyle='--')
        kw_res = dict(color='tab:red',     linewidth=0.7,
                      label='residual MF30 - K(M2)')

        ax = axs[i, 0]
        ax.plot(t_s, mf30, **kw_mf)
        ax.plot(t_s, u_fb, **kw_rec)
        ax.axvline(t_mot, color='gray', linewidth=0.8, linestyle=':', label='motion start')
        ax.set_ylabel(f'{aname} [A]')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title('Full trajectory (iter0)', fontsize=9)

        ax2 = axs[i, 1]
        ax2.plot(t_zoom, mf30[mot_idx:zoom_end], **kw_mf)
        ax2.plot(t_zoom, u_fb[mot_idx:zoom_end], **kw_rec)
        ax2.set_ylabel(f'{aname} [A]')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        if i == 0:
            ax2.set_title('Zoom: first 200 ms of motion\n(filter warm-up visible here)', fontsize=9)

        ax3 = axs[i, 2]
        ax3.plot(t_s, resid, **kw_res)
        ax3.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax3.axvline(t_mot, color='gray', linewidth=0.8, linestyle=':')
        ax3.set_ylabel(f'{aname} [A]')
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        if i == 0:
            ax3.set_title('Residual (MF30 - K(M2))', fontsize=9)

    for col in range(3):
        axs[-1, col].set_xlabel('Time [s]')

    fig.suptitle(
        'Check 4: iter0 CLOE consistency  (feedforward OFF -> MF30 = K(M2))\n'
        'Good match confirms unit chain; residual shows filter warm-up and model mismatch.\n'
        'Vertical dotted line = motion start',
        fontsize=9
    )
    fig.tight_layout()
    path = os.path.join(_SAVE_DIR, 'check4_iter0_cloe_consistency.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\n  Figure saved: {path}')
    plt.close(fig)


# -- Main ---------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(_SAVE_DIR, exist_ok=True)

    # Load available log files
    print('Loading log files...')
    data_dict = {}
    for name, fname in _ITER_SPECS:
        path = os.path.join(_DATA_ROOT, fname)
        if not os.path.isfile(path):
            print(f'  {name:<12}: not found at {path}')
            continue
        try:
            d = _load_log_raw(path)
            data_dict[name] = d
            print(f'  {name:<12}: {d["T"]} samples  ({d["T"] / d["fs"]:.2f} s)')
        except Exception as exc:
            print(f'  {name:<12}: ERROR -- {exc}')

    if not data_dict:
        print('No log files found. Check OP_FOLDER and _DATA_ROOT.')
        sys.exit(1)

    # Load controller filters from Telica.mat
    print('\nLoading controller filters from Telica.mat...')
    try:
        filters = _load_filters()
        enc_per_um = 1e-6 / filters['enc_res']
        print(f'  AmplifierGain : {filters["amp_gain"]:.10f} A/ci')
        print(f'  EncoderRes    : {filters["enc_res"]:.4e} m/count  =>  1 um = {enc_per_um:.1f} counts')
        for axis in ('X', 'Y'):
            (b1, a1), (b2, a2) = filters[axis]
            print(f'  {axis} Filter1: order {len(b1)-1}  b[:3]={np.round(b1[:3], 3).tolist()}'
                  f'  a[:3]={np.round(a1[:3], 3).tolist()}')
            print(f'  {axis} Filter2: order {len(b2)-1}  b[:3]={np.round(b2[:3], 3).tolist()}'
                  f'  a[:3]={np.round(a2[:3], 3).tolist()}')
    except Exception as exc:
        print(f'  ERROR loading filters: {exc}')
        filters = None

    # Run the four checks
    check1_iter0_sanity(data_dict)
    check2_ff_evolution(data_dict)
    if filters is not None:
        check3_filter_reconstruction(data_dict, filters)
        check4_iter0_cloe_consistency(data_dict, filters)
    else:
        print('\n[Check 3/4] Skipped -- filter load failed.')

    print(f'\nDiagnostic complete. All figures in:\n  {_SAVE_DIR}')
