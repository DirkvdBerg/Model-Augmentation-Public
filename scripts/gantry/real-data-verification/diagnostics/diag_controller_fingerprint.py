"""
diag_controller_fingerprint.py
------------------------------
Fingerprint check of the REAL Telica controllers (dFeedbackControllersTelica.mat,
axis order confirmed: 1 LX1, 2 LX2, 3 LY, 4 RX1, 5 RX2, 6 RY) against the
empirical controller FRF measured from iter0 (M2 [m] -> MF230 [A], exact
input/output pair of the feedback controller since feedforward = 0).

Questions this answers per BHL axis (X1 -> LX1, X2 -> LX2, Y -> LY):
  1. Log rate: the controllers are designed at 20 kHz. If the log is at 20 kHz
     the empirical FRF matches K(e^{j 2 pi nu}); if the log is a 10 kHz
     decimation it matches K(e^{j pi nu}). Notch alignment decides.
  2. Scale: LS flat gain between empirical and controller response in the
     coherent band. 1.0 = exact.
  3. Shape: residual after flat gain. Structured deviation = extra dynamics
     (e.g. cross-coupling / decoupling transform) not captured by SISO K.

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_controller_fingerprint.py
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
_CTRL_BA   = os.path.join(_ROOT, 'kamtin-data', 'dFeedbackControllersTelica_ba.mat')
_SAVE_DIR  = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                          'diagnostics', 'controller_fingerprint')

FS_CTRL  = 20_000.0
NPERSEG  = 4096          # HEURISTIC: ~7 averaged segments on ~17k samples
COH_MIN  = 0.9           # HEURISTIC: fit band requires coherent excitation
NU_MIN   = 8.0 / NPERSEG
NU_MAX   = 0.45

# BHL axis -> controller row (0-based) per Quinten: 1 LX1, 2 LX2, 3 LY
_MAP = (('GTRX1', 'X1', 0, 'LX1'),
        ('GTRX2', 'X2', 1, 'LX2'),
        ('GTRY',  'Y',  2, 'LY'))


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


def load_controllers():
    m = loadmat(_CTRL_BA, squeeze_me=True)
    num_mat, den_mat = np.atleast_2d(m['num_mat']), np.atleast_2d(m['den_mat'])
    ctrls = []
    for i in range(num_mat.shape[0]):
        b = np.trim_zeros(num_mat[i], 'b')
        a = np.trim_zeros(den_mat[i], 'b')
        ctrls.append((b, a))
    return ctrls, float(m['Ts'])


def fit_scale(H_emp, H_mod, mask):
    logdiff = np.log10(np.abs(H_emp[mask])) - np.log10(np.abs(H_mod[mask]))
    return 10.0 ** np.mean(logdiff), 20.0 * np.std(logdiff)


def main():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    data = load_log(os.path.join(_DATA_ROOT, 'iter0.log'))
    ctrls, Ts = load_controllers()
    assert abs(Ts - 1.0 / FS_CTRL) < 1e-12

    print('=' * 74)
    print('CONTROLLER FINGERPRINT: iter0 empirical FRF vs real Telica controllers')
    print('mapping: X1 -> LX1 (row 1), X2 -> LX2 (row 2), Y -> LY (row 3)')
    print('=' * 74)

    summary = {}
    for code, name, row, ctrl_name in _MAP:
        e  = data[f'BHL_{code}_M2'].to_numpy(float) * 1e-6    # m
        i_ = data[f'BHL_{code}_MF230'].to_numpy(float)        # A
        e, i_ = e - e.mean(), i_ - i_.mean()

        # THEORY: H1 = Syx/Sxx FRF estimator (Pintelon & Schoukens 2012, Ch. 2)
        nu, Pxy = csd(e, i_, fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        _,  Pxx = welch(e,  fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        _,  coh = coherence(e, i_, fs=1.0, nperseg=NPERSEG, noverlap=NPERSEG // 2)
        H_emp = Pxy / Pxx                                     # A/m

        b, a = ctrls[row]
        _, H20 = freqz(b, a, worN=2 * np.pi * nu)   # hyp A: log at 20 kHz
        _, H10 = freqz(b, a, worN=np.pi * nu)       # hyp B: log at 10 kHz (nu_phys = nu/2)

        mask = (coh > COH_MIN) & (nu > NU_MIN) & (nu < NU_MAX)
        if mask.sum() < 10:
            mask = (coh > 0.7) & (nu > NU_MIN) & (nu < NU_MAX)
        sA, rA = fit_scale(H_emp, H20, mask)
        sB, rB = fit_scale(H_emp, H10, mask)
        winner = '20 kHz' if rA < rB else '10 kHz'
        print(f'\n{name} vs {ctrl_name}:')
        print(f'  log=20kHz: scale={sA:7.3f}  shape residual={rA:6.2f} dB')
        print(f'  log=10kHz: scale={sB:7.3f}  shape residual={rB:6.2f} dB')
        print(f'  -> best rate hypothesis: {winner}   n_fit={int(mask.sum())}')
        summary[name] = dict(ctrl=ctrl_name, scale_20k=sA, resid_20k_dB=rA,
                             scale_10k=sB, resid_10k_dB=rB, winner=winner)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [3, 1]})
        ax1.loglog(nu, np.abs(H_emp), 'k', lw=1.0,
                   label='empirical M2 -> MF230 (iter0) [A/m]')
        ax1.loglog(nu, sA * np.abs(H20), 'C0--', lw=1.2,
                   label=f'{ctrl_name} if log = 20 kHz, x{sA:.2f} (resid {rA:.1f} dB)')
        ax1.loglog(nu, sB * np.abs(H10), 'C3--', lw=1.2,
                   label=f'{ctrl_name} if log = 10 kHz, x{sB:.2f} (resid {rB:.1f} dB)')
        # context: the other five controllers, 20 kHz axis, thin gray
        for j, (bj, aj) in enumerate(ctrls):
            if j == row:
                continue
            _, Hj = freqz(bj, aj, worN=2 * np.pi * nu)
            ax1.loglog(nu, np.abs(Hj), color='0.85', lw=0.6, zorder=0)
        ax1.set_ylabel('|K|  [A/m]')
        ax1.set_title(f'{name}: does {ctrl_name} match the measured controller, '
                      f'and at which log rate?')
        ax1.legend(fontsize=8)
        ax1.grid(True, which='both', alpha=0.3)
        ax2.semilogx(nu, coh, 'k', lw=0.8)
        ax2.axhline(COH_MIN, color='C1', ls=':', lw=1)
        ax2.set_ylabel('coherence')
        ax2.set_xlabel('normalized frequency [cycles/sample of log]')
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        fp = os.path.join(_SAVE_DIR, f'fingerprint_{name}.png')
        fig.savefig(fp, dpi=150)
        plt.close(fig)
        print(f'  figure -> {os.path.relpath(fp, _ROOT)}')

    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\nsummary  ->', os.path.relpath(os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
