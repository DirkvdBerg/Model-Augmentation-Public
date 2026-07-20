"""
diag_controller_overlay.py
--------------------------
Decisive cross-check combining diag_log_rate.py and diag_frf_controller.py:
overlay THREE independent estimates of the feedback controller on one
physical-frequency axis, all in [dac/cnt]:

  1. K_eff from the Telica.mat FRF campaign: G^-1 (S^-1 - I), unit dac/cnt native.
  2. Empirical M2 -> MF230 FRF from iter0.log (training campaign, pure feedback),
     converted A/um -> dac/cnt via 1/amp_gain and 1/enc_per_um. Plotted twice:
     frequency axis under the 10 kHz and under the 20 kHz log-rate assumption.
  3. Documented Filter1*Filter2 at 20 kHz.

Readout:
  - If (2) overlays (1) under one rate assumption: the training-time controller
    equals the FRF-time controller, the log rate is resolved by which assumption
    aligns, and CLOE can use a controller model fitted to K_eff.
  - If (2) overlays (3): the documented filters are correct after all.
  - The 11-22x time-domain mismatch should be explained by where the M2 error
    spectrum sits relative to the (1)/(3) low-frequency divergence.

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_controller_overlay.py
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
                          'diagnostics', 'controller_overlay')

FS_CTRL  = 20_000.0
NPERSEG  = 4096            # HEURISTIC: ~7 averaged segments on ~17k samples
COH_MIN  = 0.9             # HEURISTIC: alignment metric only where coherent

_AX_CODES = ('GTRX1', 'GTRX2', 'GTRY')
_AX_NAMES = ('X1', 'X2', 'Y')
_AX_CH    = (0, 0, 1)      # FRF channel: X1/X2 -> X (0), Y -> Y (1)
_AX_FILT  = ('X', 'X', 'Y')


def load_log(path):
    with open(path, 'r') as fh:
        header = fh.readline()
    cols = [c.strip().split(':')[0].replace('.', '_')
            for c in header.rstrip('\n').split('\t')]
    seen, uniq = {}, []
    for c in cols:
        n = seen.get(c, 0)
        seen[c] = n + 1
        uniq.append(c if n == 0 else f'{c}__{n}')
    return pd.read_csv(path, sep='\t', header=None, names=uniq, skiprows=1,
                       engine='python', index_col=False).dropna(axis=1, how='all')


def load_telica():
    mat = loadmat(_TELICA, squeeze_me=True, struct_as_record=False)
    mp = mat['MachineParam']
    frf = mp.Modules.XYZ.Plant.Local.FRFMeasurement
    G = [np.asarray(frf.frfPlant[i]) for i in range(len(frf.frfPlant))]
    S = [np.asarray(frf.frfSensitivity[i]) for i in range(len(frf.frfSensitivity))]
    f = np.asarray(frf.freqsPlant, float).ravel()
    I3 = np.eye(3)
    K_eff = []
    for Gp, Sp in zip(G, S):
        Gp = np.moveaxis(Gp, -1, 0)
        Sp = np.moveaxis(Sp, -1, 0)
        K_eff.append(np.linalg.inv(Gp) @ (np.linalg.inv(Sp) - I3))

    filts = {}
    for axname in ('X', 'Y'):
        ctr = getattr(mp.Axes, axname).Controllers
        pair = []
        for fname in ('Filter1', 'Filter2'):
            v = getattr(ctr, fname).Values
            b = np.trim_zeros(np.array([getattr(v, f'b{i}') for i in range(7)], float), 'b')
            a = np.trim_zeros(np.concatenate(([1.0],
                 [getattr(v, f'a{i}') for i in range(1, 7)])).astype(float), 'b')
            pair.append((b, a))
        filts[axname] = pair
    hw = mp.Axes.X.ElectronicHardwareInfo
    amp_gain = float(hw.Motor.AmplifierGain.Value)   # A/DAC
    enc_res  = float(hw.EncoderResolution.Value)     # m/cnt
    return dict(K_eff=K_eff, f_frf=f, filters=filts,
                amp_gain=amp_gain, enc_per_um=1e-6 / enc_res)


def median_logdist(f_a, H_a, f_b, H_b, band):
    """Median |log10 ratio| between two magnitude curves on overlapping band."""
    mask = (f_a >= band[0]) & (f_a <= band[1])
    if mask.sum() < 5:
        return np.nan
    Hb_i = np.interp(f_a[mask], f_b, np.abs(H_b))
    return float(np.median(np.abs(np.log10(np.abs(H_a[mask])) - np.log10(Hb_i))))


def main():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    tel = load_telica()
    data = load_log(os.path.join(_DATA_ROOT, 'iter0.log'))

    print('=' * 70)
    print('DIAG CONTROLLER OVERLAY: iter0 empirical vs FRF-campaign K_eff '
          'vs documented F1*F2')
    print('=' * 70)

    summary = {}
    for code, name, ch, fax in zip(_AX_CODES, _AX_NAMES, _AX_CH, _AX_FILT):
        x = data[f'BHL_{code}_M2'].to_numpy(float)     # um
        y = data[f'BHL_{code}_MF230'].to_numpy(float)  # A
        x = x - x.mean()
        y = y - y.mean()

        # THEORY: H1 = Syx/Sxx FRF estimator (Pintelon & Schoukens 2012, Ch. 2)
        nu, Pxy = csd(x, y, fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        _,  Pxx = welch(x,  fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        _,  coh = coherence(x, y, fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        # convert A/um -> dac/cnt (both constants from Telica.mat)
        H_emp = (Pxy / Pxx) / tel['amp_gain'] / tel['enc_per_um']
        keep = (coh > COH_MIN) & (nu > 4.0 / NPERSEG)

        # K_eff diagonal, median magnitude across 9 positions
        K_all = np.stack([np.abs(K[:, ch, ch]) for K in tel['K_eff']])
        K_med = np.median(K_all, axis=0)
        f_frf = tel['f_frf']

        # documented filters
        w = 2 * np.pi * f_frf / FS_CTRL
        (b1, a1), (b2, a2) = tel['filters'][fax]
        _, h1 = freqz(b1, a1, worN=w)
        _, h2 = freqz(b2, a2, worN=w)
        H_doc = h1 * h2

        # alignment metric per rate hypothesis, band where iter0 has coherence
        f10, f20 = nu * 10_000.0, nu * 20_000.0
        band = (20.0, 2000.0)   # HEURISTIC: overlap band with usable coherence
        d10_k = median_logdist(f10[keep], H_emp[keep], f_frf, K_med, band)
        d20_k = median_logdist(f20[keep], H_emp[keep], f_frf, K_med, band)
        d10_d = median_logdist(f10[keep], H_emp[keep], f_frf, H_doc, band)
        d20_d = median_logdist(f20[keep], H_emp[keep], f_frf, H_doc, band)
        print(f'\n{name}: median |log10 ratio| to K_eff:   '
              f'10kHz-axis {d10_k:.3f}   20kHz-axis {d20_k:.3f}')
        print(f'{name}: median |log10 ratio| to F1*F2:   '
              f'10kHz-axis {d10_d:.3f}   20kHz-axis {d20_d:.3f}')
        summary[name] = dict(dist_Keff_10k=d10_k, dist_Keff_20k=d20_k,
                             dist_doc_10k=d10_d, dist_doc_20k=d20_d)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(f_frf, K_med, 'C2', lw=1.8,
                  label='K_eff (FRF campaign, median of 9 pos)')
        ax.loglog(f_frf, np.abs(H_doc), 'C0', lw=1.2, alpha=0.8,
                  label='Filter1*Filter2 (documented, 20 kHz)')
        ax.loglog(f10[keep], np.abs(H_emp[keep]), 'C3.', ms=3,
                  label=f'iter0 empirical, 10 kHz axis (dist {d10_k:.2f})')
        ax.loglog(f20[keep], np.abs(H_emp[keep]), 'C1.', ms=3,
                  label=f'iter0 empirical, 20 kHz axis (dist {d20_k:.2f})')
        ax.axvspan(*band, color='C2', alpha=0.05)
        ax.set_xlabel('frequency  [Hz]')
        ax.set_ylabel('|K|  [dac/cnt]')
        ax.set_title(f'{name}: which controller curve does the training data follow, '
                     f'and at which log rate?')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fp = os.path.join(_SAVE_DIR, f'overlay_{name}.png')
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f'    figure -> {os.path.relpath(fp, _ROOT)}')

    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\nsummary  ->', os.path.relpath(os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
