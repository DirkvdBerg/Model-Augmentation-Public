"""
diag_log_rate.py
----------------
Resolve the Telica .log sampling-rate ambiguity without using timestamps
(D-061 forbids them):
    Hypothesis A: log rate = 20 kHz (Telica.mat Axes.X.SamplingFrequency = 20000,
                  description "The number of samples logged per second")
    Hypothesis B: log rate = 10 kHz, 2:1 decimation of the 20 kHz DSP
                  (D-061, AccurET manual FsHz = 1/(2*PLTI))

Method (rate fingerprint):
The DSP filters Filter1*Filter2 (Telica.mat) are designed at fs_ctrl = 20 kHz,
so their features (notches, integrator slope) sit at fixed NORMALIZED
frequencies nu = f/20000. In iter0 the pair (M2 -> MF230) is an exact
input/output pair of the controller chain (feedforward = 0, Check 1 of
diag_cloe_signals.py). The empirical FRF of the logged pair lives on the
normalized axis of the LOG rate:
    under A: features appear at nu_log = nu
    under B: features appear at nu_log = 2*nu (physical f maps to f/10000)
Whichever hypothesis matches the empirical FRF shape with a single flat gain
wins; that flat gain is the residual scale factor of the documented chain
(candidate HIGS gain / undocumented unit factor).

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_log_rate.py
"""

__project_origin__ = "added"

import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import csd, welch, coherence, freqz

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))

OP_FOLDER  = 'xpos_-60_ypos-40'
_DATA_ROOT = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
                          '06 40 mm XL 80 mm YL', 'train', OP_FOLDER)
_TELICA    = os.path.join(_ROOT, 'kamtin-data', 'Telica.mat')
_SAVE_DIR  = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                          'diagnostics', 'log_rate')

FS_CTRL   = 20_000.0   # Hz, DSP rate (Telica.mat SamplingTime = 5e-5 s)
NPERSEG   = 4096       # HEURISTIC: ~7 averaged segments on a ~17k-sample log
COH_MIN   = 0.95       # HEURISTIC: only fit where excitation is coherent
NU_MIN    = 8.0 / NPERSEG   # HEURISTIC: skip lowest FFT bins (leakage)
NU_MAX    = 0.45            # HEURISTIC: stay below Nyquist edge

_AX_CODES = ('GTRX1', 'GTRX2', 'GTRY')
_AX_NAMES = ('X1', 'X2', 'Y')
_AX_FILT  = ('X', 'X', 'Y')   # which Telica controller each axis uses


def load_log(path):
    """Load one iter*.log at native rate, no trimming, no resampling."""
    with open(path, 'r') as fh:
        header = fh.readline()
    cols = [c.strip().split(':')[0].replace('.', '_')
            for c in header.rstrip('\n').split('\t')]
    # make names unique (trailing empties)
    seen, uniq = {}, []
    for c in cols:
        n = seen.get(c, 0)
        seen[c] = n + 1
        uniq.append(c if n == 0 else f'{c}__{n}')
    raw = pd.read_csv(path, sep='\t', header=None, names=uniq, skiprows=1,
                      engine='python', index_col=False)
    return raw.dropna(axis=1, how='all')


def load_filters():
    """Filter1/Filter2 coefficients + unit constants from Telica.mat."""
    mat = loadmat(_TELICA, squeeze_me=True, struct_as_record=False)
    mp = mat['MachineParam']
    out = {}
    for axname in ('X', 'Y'):
        ctr = getattr(mp.Axes, axname).Controllers
        filts = []
        for fname in ('Filter1', 'Filter2'):
            v = getattr(ctr, fname).Values
            b = np.array([getattr(v, f'b{i}') for i in range(7)], float)
            a = np.concatenate(([1.0],
                 [getattr(v, f'a{i}') for i in range(1, 7)])).astype(float)
            b = np.trim_zeros(b, 'b')
            a = np.trim_zeros(a, 'b')
            filts.append((b, a))
        out[axname] = filts
    hw = mp.Axes.X.ElectronicHardwareInfo
    out['amp_gain'] = float(hw.Motor.AmplifierGain.Value)      # A/DAC
    out['enc_res']  = float(hw.EncoderResolution.Value)        # m/cnt
    return out


def chain_response(filters, axname, w):
    """|H| of the full documented chain at rad/sample vector w.
    Chain: um -> cnt (x enc_per_um) -> Filter1 -> Filter2 -> DAC -> A (x amp_gain)."""
    enc_per_um = 1e-6 / filters['enc_res']
    (b1, a1), (b2, a2) = filters[axname]
    _, h1 = freqz(b1, a1, worN=w)
    _, h2 = freqz(b2, a2, worN=w)
    return enc_per_um * h1 * h2 * filters['amp_gain']


def filter_feature_table(filters):
    """Print zeros/poles of each filter as normalized and physical frequencies."""
    print('Filter feature table (root angles as frequencies, fs_ctrl = 20 kHz):')
    for axname in ('X', 'Y'):
        for k, (b, a) in enumerate(filters[axname], start=1):
            for kind, poly in (('zero', b), ('pole', a)):
                r = np.roots(poly)
                for root in r:
                    nu = abs(np.angle(root)) / (2 * np.pi)
                    if nu < 1e-6:
                        continue  # real-axis roots carry no frequency landmark
                    print(f'  {axname} Filter{k} {kind}: |r|={abs(root):.4f}  '
                          f'nu={nu:.5f}  f_phys={nu * FS_CTRL:8.1f} Hz  '
                          f'(appears at {2 * nu:.5f} on a 10 kHz log axis)')
    print()


def fit_scale(nu, H_emp, coh, filters, axname, hypothesis):
    """LS flat-gain fit of |H_emp| against the chain response under one
    rate hypothesis. Returns (scale, residual_dB, mask)."""
    if hypothesis == 'A':          # log at 20 kHz: nu_log = nu_ctrl
        w = 2 * np.pi * nu
    else:                          # log at 10 kHz: nu_log = 2*nu_ctrl
        w = np.pi * nu
    H_mod = np.abs(chain_response(filters, axname, w))
    mask = (coh > COH_MIN) & (nu > NU_MIN) & (nu < NU_MAX) & (H_mod > 0)
    if mask.sum() < 10:
        mask = (coh > 0.8) & (nu > NU_MIN) & (nu < NU_MAX) & (H_mod > 0)
    logdiff = np.log10(np.abs(H_emp[mask])) - np.log10(H_mod[mask])
    scale = 10.0 ** np.mean(logdiff)                 # flat gain, magnitude LS in log domain
    resid_db = 20.0 * np.std(logdiff)                # shape misfit after flat gain
    return scale, resid_db, mask, H_mod


def main():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    filters = load_filters()

    print('=' * 70)
    print('DIAG LOG RATE: 20 kHz (A) vs 10 kHz decimated (B), no timestamps')
    print('=' * 70)

    # --- 0. Row counts of every log in the folder (relative-rate hint) ---
    print('\nRow counts per log file in', OP_FOLDER)
    for fn in sorted(os.listdir(_DATA_ROOT)):
        if fn.endswith('.log'):
            with open(os.path.join(_DATA_ROOT, fn), 'r') as fh:
                n = sum(1 for _ in fh) - 1
            print(f'  {fn:20s} {n:8d} rows')

    filter_feature_table(filters)

    # --- 1. Empirical FRF M2 -> MF230 from iter0 ---
    data = load_log(os.path.join(_DATA_ROOT, 'iter0.log'))
    summary = {}
    for code, name, filt_ax in zip(_AX_CODES, _AX_NAMES, _AX_FILT):
        x = data[f'BHL_{code}_M2'].to_numpy(float)     # um
        y = data[f'BHL_{code}_MF230'].to_numpy(float)  # A
        x = x - x.mean()
        y = y - y.mean()

        # THEORY: H1 = Syx/Sxx FRF estimator (Pintelon & Schoukens 2012, Ch. 2)
        nu, Pxy = csd(x, y, fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        _,  Pxx = welch(x,  fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        _,  coh = coherence(x, y, fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        H_emp = Pxy / Pxx

        sA, rA, mA, HmodA = fit_scale(nu, H_emp, coh, filters, filt_ax, 'A')
        sB, rB, mB, HmodB = fit_scale(nu, H_emp, coh, filters, filt_ax, 'B')
        winner = 'A (20 kHz)' if rA < rB else 'B (10 kHz)'
        print(f'{name}:  A(20k): scale={sA:8.3f} resid={rA:6.2f} dB   '
              f'B(10k): scale={sB:8.3f} resid={rB:6.2f} dB   -> {winner}')
        summary[name] = dict(scale_A=sA, resid_A_dB=rA,
                             scale_B=sB, resid_B_dB=rB, winner=winner,
                             n_fit_bins=int(mA.sum()))

        # --- falsifiable plot: measurement vs both scaled hypotheses ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
        ax1.loglog(nu, np.abs(H_emp), 'k', lw=1.0, label='empirical M2->MF230 (iter0)')
        ax1.loglog(nu, sA * HmodA, 'C0--', lw=1.2,
                   label=f'hyp A: 20 kHz log, gain {sA:.2f}, resid {rA:.1f} dB')
        ax1.loglog(nu, sB * HmodB, 'C3--', lw=1.2,
                   label=f'hyp B: 10 kHz log, gain {sB:.2f}, resid {rB:.1f} dB')
        ax1.set_ylabel('|H|  [A/um]')
        ax1.set_title(f'{name}: which rate hypothesis matches the measured '
                      f'controller FRF shape?')
        ax1.legend(fontsize=8)
        ax1.grid(True, which='both', alpha=0.3)
        ax2.semilogx(nu, coh, 'k', lw=0.8)
        ax2.axhline(COH_MIN, color='C1', ls=':', lw=1)
        ax2.set_ylabel('coherence')
        ax2.set_xlabel('normalized frequency  [cycles/sample of log]')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fp = os.path.join(_SAVE_DIR, f'rate_fingerprint_{name}.png')
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f'    figure -> {os.path.relpath(fp, _ROOT)}')

    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\nsummary  ->', os.path.relpath(os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
