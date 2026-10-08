"""Black/grey figures for the 08-10 deck: live5 right panel (live ResNet residual ASD), prob redraw, backup val curve.

live5: e = y - y_hat on Y, closed-loop free run over the full validation records (resid_*.npz from eval_resid.py:
baseline = model before training, plain = SSE_Interconnect_Composed_88532_best.pth, with the method =
SSE_Interconnect_Composed_88531_best.pth). Welch ASD, Hann, 4096 samples, 50 % overlap, fs 4 kHz, per record, PSD
power-averaged over the four multisine validation records. Band RMS 230-297 Hz from the same averaged PSD.
"""
__project_origin__ = "added"

import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 11, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e0e0e0', 'grid.linewidth': 0.6})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figures')
LOGS = os.path.join(HERE, '..', '..', 'server', 'initial-g')
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..', '..'))
FS = 4000.0
SHADE = '#d9d9d9'
GREY = '#8a8a8a'
SIZE = dict(figsize=(5.2667, 4.0), dpi=150)        # 790 x 600 px
STY = {'baseline': dict(color='black', ls='-', label='baseline'),
       'plain': dict(color='black', ls='--', label='plain S-DP'),
       'ps2': dict(color=GREY, ls='-', label='with the method')}


def live5(sp):
    asd, band = {}, {}
    for k in ('baseline', 'plain', 'ps2'):
        d = np.load(os.path.join(sp, 'resid_%s.npz' % k))
        names = [str(n).split('_')[0] for n in d['names']]
        ms = [i for i, n in enumerate(names) if not n.startswith('VA-T')]
        assert [names[i] for i in ms] == ['VA-S1', 'VA-Y1', 'VA-P1', 'VA-L1'], names
        f, P = welch(d['e'][ms, :, 2], fs=FS, window='hann', nperseg=4096, noverlap=2048, axis=1)
        Pm = P.mean(0)                                  # power average over the four records
        asd[k] = np.sqrt(Pm)
        sel = (f >= 230) & (f <= 297)
        band[k] = float(np.sqrt(Pm[sel].sum() * (f[1] - f[0])))
    m = (f >= 10) & (f <= 2000)
    fig, ax = plt.subplots(**SIZE)
    ax.axvspan(230, 297, color=SHADE, lw=0, zorder=0)
    for k in ('baseline', 'plain', 'ps2'):
        ax.loglog(f[m], asd[k][m], lw=1.6, **STY[k])
    ax.set_xlim(10, 2000)
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('Y residual [m/$\\sqrt{\\mathrm{Hz}}$]')
    ax.legend(loc='lower left', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'live5_right_live_resnet_Y_asd.png'))
    fig.savefig(os.path.join(OUT, 'live5_right_live_resnet_Y_asd.svg'))
    plt.close(fig)
    json.dump(dict(band_rms_230_297_m=band, checkpoints=dict(plain='SSE_Interconnect_Composed_88532_best.pth',
                                                             ps2='SSE_Interconnect_Composed_88531_best.pth')),
              open(os.path.join(OUT, 'live5_right_band_rms.json'), 'w'), indent=1)
    print('live5 right, band RMS 230-297 Hz from the averaged PSD [m]:', band)


def prob():
    """Redraw of the slide 3 figure (meeting-01-10 fig_absorber_issue.py, run 42) in black and grey."""
    d = np.load(os.path.join(REPO, 'scripts', 'gantry', 'absorber-learning-diagnosis', 'outputs', 'f_above_pole_86894.npz'))
    m = (d['f'] >= 107) & (d['f'] <= 296)
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.axvspan(230, 297, color=SHADE, lw=0, zorder=0)
    ax.semilogy(d['f'][m], np.sqrt(d['Ptt'][m]), color='black', lw=2.2, label='baseline error')
    ax.semilogy(d['f'][m], np.sqrt(d['Pee'][m]), color=GREY, lw=2.2, label='error after training')
    ax.set_xlim(106, 297)
    ax.set_ylim(2e-7, 8e-6)
    ax.set_yticks([3e-7, 1e-6, 3e-6])
    ax.set_yticklabels(['$3{\\cdot}10^{-7}$', '$10^{-6}$', '$3{\\cdot}10^{-6}$'])
    ax.minorticks_off()
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('Y error [m/$\\sqrt{\\mathrm{Hz}}$]')
    ax.legend(loc='upper left', frameon=False)
    for x, t in [(168, '90 % removed'), (263, '22 %\nremoved')]:
        ax.text(x, 2.6e-7, t, ha='center', va='bottom', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'prob_Y_error_spectrum_bw.png'), dpi=200)
    plt.close(fig)


def val_curve(path):
    txt = open(path, encoding='utf-8', errors='replace').read().replace('\r', '\n')
    v0 = float(re.search(r'Initial Validation sim-RMS= ([0-9.e+-]+)', txt).group(1))
    pts = [(int(a), float(b)) for a, b in re.findall(r'\nIt (\d+), sqrt loss [0-9.e+-]+, Val sim-RMS ([0-9.e+-]+)', txt)]
    return np.array([(0, v0)] + pts)


def backup():
    cp = val_curve(os.path.join(LOGS, 'thesis-tcompare-resnet_88531_161.out'))
    cs = val_curve(os.path.join(LOGS, 'thesis-tcompare-resnet_88531_162.out'))
    fig, ax = plt.subplots(**SIZE)
    ax.semilogy(cp[:, 0], cp[:, 1], 'o--', color='black', ms=4, label='plain S-DP')
    ax.semilogy(cs[:, 0], cs[:, 1], 'o-', color=GREY, ms=4, label='with the method')
    ax.axvline(2889, color='black', ls=':', lw=1.2)
    ax.text(3200, cs[-1, 1], 'NaN at update 2,889,\nbest weights kept', fontsize=11, va='center')
    ax.set_xlabel('update')
    ax.set_ylim(2.5e-6, 2.2e-5)
    ax.set_ylabel('validation error, full records [m]')
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'backup_live_resnet_val_curve_bw.png'))
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    live5(sys.argv[1])
    prob()
    backup()
