"""F2: why the band is 106 to 297 Hz.
Top: closed-loop FRF Y from F_Y, truth (BLA, K1, Y = 0) vs nominal baseline.
Bottom: cumulative share of the resolved FRF difference above the 106 Hz crossover, all 9 stage entries, mean over
the five training Y; the band ends where the share reaches 90 %. Construction as
excitation-closed-loop/j7_band.py (T1: |E| > 2.45 sigma; T2: weight sum over entries of resolved mean |E|^2),
on the tanh-truth BLA band_tanh_g1000/j2_bla.npz."""
__project_origin__ = "added"

import os
import sys

import numpy as np

from style import plt, BAND, BLUE, ORANGE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..', '..'))
EX = os.path.join(REPO, 'scripts', 'gantry', 'excitation-closed-loop')
SRC = os.path.join(REPO, 'scripts', 'gantry', 'coulomb-tanh-gain', 'outputs', 'band_tanh_g1000', 'j2_bla.npz')
OUT = os.path.join(HERE, '..', 'figures')
sys.path.insert(0, EX)
import common as C  # noqa: E402

YS = [-0.30, -0.15, 0.0, 0.15, 0.30]
FC = 106.5


def main():
    os.makedirs(OUT, exist_ok=True)
    j2 = np.load(SRC)
    f = j2['fl_K1_Y+0.00'][:, 1]
    K1 = C.ctrl_frf(0.0, f)
    E, S2 = [], []
    for y in YS:
        tag = 'K1_Y%+.2f' % y
        PSt = j2['PS_' + tag] @ C.P
        _, PSb = C.loop(C.frf_zoh(*C.base_mck(C.THETA0, y), f), K1)
        E.append(PSt - PSb)
        S2.append(np.real(j2['vt_' + tag]) @ (C.P ** 2))
        if y == 0.0:
            PSt0, PSb0 = PSt[:, 2, 2], PSb[:, 2, 2]
    Em2 = np.mean(np.abs(np.array(E)) ** 2, axis=0)
    resolved = np.sqrt(Em2) > 2.45 * np.sqrt(np.mean(S2, axis=0))
    w = np.sum(Em2 * resolved, axis=(1, 2))
    above = f >= FC
    cum = np.cumsum(w * above) / np.sum(w * above)
    f90 = f[np.argmax(cum >= 0.90)]
    print('cumulative share from %.1f Hz reaches 90 %% at %.1f Hz' % (FC, f90))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.4, 4.9), sharex=True)
    for ax in (a1, a2):
        ax.axvspan(106, 297, color=BAND, lw=0, zorder=0)
        ax.axvline(106, color='black', ls='-.', lw=1.2)
    a1.loglog(f, np.abs(PSt0), color='black', label='truth')
    a1.loglog(f, np.abs(PSb0), color=BLUE, ls='--', label='baseline')
    a1.set_ylabel('|FRF| [m/N]')
    a1.legend(loc='lower left', frameon=False)
    a1.annotate('150 Hz', xy=(150, 1.7e-8), xytext=(40, 1.2e-8), fontsize=12, arrowprops=dict(arrowstyle='->'))
    a1.annotate('264 Hz', xy=(264, 2.4e-7), xytext=(400, 4e-7), fontsize=12, arrowprops=dict(arrowstyle='->'))
    a2.semilogx(f[above], 100 * cum[above], color=ORANGE, lw=2.5)
    a2.axhline(90, color='black', lw=1.0, ls='--')
    a2.plot([f90], [90], 'o', color='black')
    a2.text(f90 * 1.08, 80, '90 %% at %.0f Hz' % f90, fontsize=12, va='top')
    a2.text(100, 50, 'crossover\n106 Hz', ha='right', va='center', fontsize=12)
    a2.set_ylim(0, 105)
    a2.set_ylabel('share of the\ndifference [%]')
    a2.set_xlabel('frequency [Hz]')
    a2.set_xlim(10, 1e3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F2_frf_band.png'), dpi=200)


if __name__ == '__main__':
    main()
