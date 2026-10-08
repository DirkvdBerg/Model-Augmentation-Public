"""F4: Y channel, validation multisine records, run 42 (absorber-learning-diagnosis/outputs/f_above_pole_86894.npz).
Black: error of the baseline at true parameters (T = y_truth - y_baseline); orange: error of the trained
augmented model. Error removed per band (90 / 89 / 22 %) from DIAGNOSIS.md (E1)."""
__project_origin__ = "added"

import os

import numpy as np

from style import plt, BAND, BAD, ORANGE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..', '..'))
SRC = os.path.join(REPO, 'scripts', 'gantry', 'absorber-learning-diagnosis', 'outputs', 'f_above_pole_86894.npz')
OUT = os.path.join(HERE, '..', 'figures')


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(SRC)
    m = (d['f'] >= 107) & (d['f'] <= 296)
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.axvspan(106, 230, color=BAND, lw=0, zorder=0)
    ax.axvspan(230, 297, color=BAD, lw=0, zorder=0)
    ax.semilogy(d['f'][m], np.sqrt(d['Ptt'][m]), color='black', lw=2.2, label='baseline error')
    ax.semilogy(d['f'][m], np.sqrt(d['Pee'][m]), color=ORANGE, lw=2.2, label='error after training')
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
    fig.savefig(os.path.join(OUT, 'F4_absorber_issue.png'), dpi=200)


if __name__ == '__main__':
    main()
