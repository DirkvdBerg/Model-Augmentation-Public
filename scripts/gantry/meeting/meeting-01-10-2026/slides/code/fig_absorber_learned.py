"""F4: the absorber is learned below its 212 Hz pole, not above it (runs 42, 44, 48).

|H| = |S_LT / S_TT|, learned correction over the correction the truth needs, Y channel, four multisine
validation records (absorber-learning-diagnosis/f_above_pole.py). |H| = 1 is a perfect correction.
Nothing recomputed differently, only drawn.
Run: conda run -n GraduationProject python fig_absorber_learned.py
"""
__project_origin__ = "added"

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..', '..'))
SRC = os.path.join(REPO, 'scripts', 'gantry', 'absorber-learning-diagnosis', 'outputs')
OUT = os.path.join(HERE, '..', 'figures')

RUNS = [('86894', 'run 42 (U, 2 states)', '#2a78d6', '-'),
        ('86918', 'run 44 (U, 2 states, seed 2)', '#d9581f', '--'),
        ('87018', 'run 48 (OBC, 4 states)', '#6a3fb0', ':')]


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {j: np.load(os.path.join(SRC, 'f_above_pole_%s.npz' % j)) for j, *_ in RUNS}
    fig, a1 = plt.subplots(figsize=(7.5, 4.0))
    for j, lab, c, ls in RUNS:
        d = data[j]
        m = (d['f'] >= 106) & (d['f'] <= 297)                 # multisine band only; outside it H is not excited
        a1.plot(d['f'][m], np.abs(d['H'][m]), color=c, ls=ls, lw=1.6, label=lab)
    a1.axhline(1.0, color='black', lw=1.0)
    a1.text(297, 1.03, 'perfect correction', ha='right', va='bottom', fontsize=8)
    for fm, t in [(150, 'anti-res.\n150'), (212, 'pole\n212'), (264, 'CL peak\n264')]:
        a1.axvline(fm, color='#b0b0b0', ls=':', lw=1.0)
        a1.text(fm, 1.32, t, ha='center', va='top', fontsize=7.5, color='#505050')
    a1.set_xlim(106, 297)
    a1.set_ylim(0, 1.35)
    a1.set_xlabel('frequency [Hz]  (multisine band 106 to 297 Hz)')
    a1.set_ylabel('|learned / needed correction|  [-]')
    a1.set_title('Learned vs needed absorber correction (Y)', fontsize=10)
    a1.legend(loc='lower left', fontsize=8, frameon=False)
    a1.grid(True, axis='y', color='#e0e0e0', lw=0.5)
    for sp in ('top', 'right'):
        a1.spines[sp].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, 'F4_absorber_learned.png')
    fig.savefig(p, dpi=200)
    print('saved', p)


if __name__ == '__main__':
    main()
