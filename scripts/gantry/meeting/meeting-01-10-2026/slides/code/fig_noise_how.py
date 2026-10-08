"""F1b: how the injected encoder noise is shaped: injected / measured amplitude per stage axis.
Injected: v_enc averaged over every noisy record at Y_op 0 (independent draws of one design). Measured: Telica standstill servo error
(coulomb-tanh-gain/outputs/noise_shape_Y0.mat, phi_meas). Both power spectra averaged in 1/6-octave bands
(Welch Hann 32768 first, 0.6 Hz lines), ratio = sqrt of the band powers. By design 1/|S| above 50 Hz and 1 below."""
__project_origin__ = "added"

import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from style import plt, BLUE, ORANGE, PURPLE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..', '..'))
CHECK = os.path.join(REPO, 'scripts', 'gantry', 'coulomb-tanh-gain', 'outputs', 'noise_shape_Y0.mat')
REC = os.path.join(REPO, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD', 'Training', 'TR-S3_r1.mat')
OUT = os.path.join(HERE, '..', 'figures')


def main():
    os.makedirs(OUT, exist_ok=True)
    m = loadmat(CHECK, squeeze_me=True)
    fm, pm, fc = m['f_meas'], m['phi_meas'], float(m['fc'])
    # every noisy record at Y_op 0 carries an independent draw of the same noise design: average them
    root = os.path.dirname(os.path.dirname(REC))
    import csv
    files = [os.path.join(root, r['folder'], r['file'] + '.mat')
             for r in csv.DictReader(open(os.path.join(root, 'MANIFEST.csv')))
             if abs(float(r['Y_op'])) < 1e-9 and r['block'] != 'frf']
    Pv = 0
    for p in files:
        v = loadmat(p, variable_names=['v_enc'])['v_enc']
        f, P = welch(v, fs=20000, window='hann', nperseg=32768, noverlap=16384, axis=0)
        Pv = Pv + P / len(files)
    print('averaged %d noise draws at Y_op 0' % len(files))
    edges = 20 * 2 ** (np.arange(0, 60) / 6)
    edges = edges[edges <= 1e4]
    fc_b = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.axvspan(15, fc, color='#ececec', lw=0, zorder=0)
    for j, (n, c) in enumerate([('X1', BLUE), ('X2', PURPLE), ('Y', ORANGE)]):
        pmi = np.interp(f, fm, pm[:, j])
        r = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (f >= lo) & (f < hi)
            r.append(np.sqrt(Pv[s, j].sum() / pmi[s].sum()) if s.any() else np.nan)
        ax.semilogx(fc_b, r, color=c, lw=2.8, label=n)
    ax.axhline(1.0, color='black', lw=1.2)
    ax.set_xlim(15, 1e4)
    ax.set_ylim(0.7, 1.45)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('injected / measured noise')
    ax.legend(loc='upper right', ncol=3, frameon=False)
    ax.text(27, 0.74, 'not\ncorrected', ha='center', va='bottom', fontsize=13)
    ax.text(90, 1.33, 'inject\nmore', ha='center', va='bottom', fontsize=13)
    ax.text(380, 0.74, 'inject\nless', ha='center', va='bottom', fontsize=13)
    ax.text(3000, 0.74, 'unchanged', ha='center', va='bottom', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F1b_noise_how.png'), dpi=200)


if __name__ == '__main__':
    main()
