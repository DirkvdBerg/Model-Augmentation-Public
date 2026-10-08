"""F3: friction force against velocity on the X1 rail (cc = 16.8 N): ideal Coulomb step, cc tanh(g v) at g = 1000
(used) and g = 100; shaded: rail velocity under the multisine (7e-3 to 8e-3 m/s rms, D-226), both directions."""
__project_origin__ = "added"

import os

import numpy as np

from style import plt, BLUE, GREY, BAD

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
CC = 16.8                                                        # [N], X1 rail (Garcia)


def main():
    os.makedirs(OUT, exist_ok=True)
    v = np.linspace(-1.5e-2, 1.5e-2, 2001)
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    for s in (-1, 1):
        ax.axvspan(s * 7e-3, s * 8e-3, color=BAD, lw=0, zorder=0)
    ax.plot(v, CC * np.sign(v), color='black', lw=2.0, label='Coulomb')
    ax.plot(v, CC * np.tanh(100 * v), color=GREY, ls='--', label='tanh, g = 100')
    ax.plot(v, CC * np.tanh(1000 * v), color=BLUE, lw=3.0, label='tanh, g = 1000 (used)')
    ax.axhline(0, color='#b0b0b0', lw=0.8)
    ax.set_xlim(-1.5e-2, 1.5e-2)
    ax.set_ylim(-20, 20)
    ax.set_xticks([-1e-2, -5e-3, 0, 5e-3, 1e-2])
    ax.set_xticklabels(['$-10^{-2}$', '$-5{\\cdot}10^{-3}$', '0', '$5{\\cdot}10^{-3}$', '$10^{-2}$'])
    ax.set_xlabel('rail velocity [m/s]')
    ax.set_ylabel('friction force X1 [N]')
    ax.legend(loc='upper left', frameon=False)
    ax.text(8.5e-3, -8, 'rail velocity\nunder multisine', ha='left', va='center', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F3_friction_tanh.png'), dpi=200)


if __name__ == '__main__':
    main()
