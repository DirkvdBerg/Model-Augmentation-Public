"""D-223 figure: servo error of the Karnopp truth and the tanh truths, noise-free and noisy.

Reads outputs/sweep_<stem>_<version>.mat written by run_tanh_sweep.m and draws one PNG:
rows X1, X2, Y; columns noise-free, noisy; a window of about 1.2 s holding moves and dwells.
The error is a rolling rms on a log axis, so a sustained micrometre limit cycle and a decaying
nanometre transient are both readable on one scale. Moves (reference changing) are shaded.

Run: conda run -n GraduationProject python plot_tanh_sweep.py [stem]
"""
__project_origin__ = "added"

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
STEM = sys.argv[1] if len(sys.argv) > 1 else 'TR-T1_r1'
WIN_S = 0.010        # HEURISTIC: 10 ms rolling rms, a fifth of the ~50 ms hunting period
FLOOR = 1e-12        # m, plotting floor for a settled noise-free error (round-off level)
KARNOPP = '#000000'
BLUES = ['#a9c8ef', '#6fa3e3', '#2a78d6', '#1a55a3', '#0d3470']   # one hue, light to dark


def rolling_rms(x, n):
    c = np.cumsum(np.concatenate([np.zeros((1, x.shape[1])), x.astype(np.float64) ** 2]), axis=0)
    r = np.sqrt(np.maximum(c[n:] - c[:-n], 0.0) / n)
    return np.concatenate([np.full((n - 1, x.shape[1]), np.nan), r])


def load(version):
    p = os.path.join(OUT, f'sweep_{STEM}_{version}.mat')
    if not os.path.isfile(p):
        return None
    m = loadmat(p, squeeze_me=True)
    return dict(E=m['E'], names=[str(s) for s in np.atleast_1d(m['names'])],
                mov=m['mov'].astype(bool), dw=np.atleast_2d(m['dw']).astype(int), fs=float(m['fs']))


def main():
    data = {v: load(v) for v in ('noise-free', 'noisy')}
    cols = [v for v in data if data[v] is not None]
    ref = data[cols[0]]
    fs, dw = ref['fs'], ref['dw']
    a = dw[min(2, len(dw) - 1), 0] - 1                   # third scored dwell (MATLAB 1-based)
    i0 = max(0, a - int(0.2 * fs))
    i1 = min(len(ref['mov']), i0 + int(1.2 * fs))
    t = np.arange(i0, i1) / fs
    n = int(round(WIN_S * fs))

    fig, axes = plt.subplots(3, len(cols), figsize=(6.2 * len(cols), 7.2), sharex=True,
                             squeeze=False)
    for j, v in enumerate(cols):
        d = data[v]
        mv = d['mov'][i0:i1]
        for i, axis in enumerate(['X1', 'X2', 'Y']):
            ax = axes[i, j]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], mv.astype(int), [0]])))
            for s, e in zip(edges[0::2], edges[1::2]):
                ax.axvspan(t[s], t[e - 1], color='#e6e6e6', lw=0, zorder=0)
            for k, name in enumerate(d['names']):
                y = rolling_rms(d['E'][:, i, k][:, None], n)[i0:i1, 0]
                col = KARNOPP if name == 'Karnopp' else BLUES[(k - 1) % len(BLUES)]
                lw = 1.6 if name == 'Karnopp' else 1.2
                ax.semilogy(t, np.maximum(y, FLOOR), color=col, lw=lw, label=name, zorder=2)
            ax.set_ylim(FLOOR, 1e-3)
            ax.grid(True, which='major', color='#d9d9d9', lw=0.5)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if j == 0:
                ax.set_ylabel(f'{axis} servo error\n{int(WIN_S * 1e3)} ms rms [m]')
            if i == 0:
                ax.set_title(f'{v}', fontsize=11)
        axes[-1, j].set_xlabel('t [s]   (grey: reference moving)')
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=len(lab), frameon=False,
               fontsize=9)
    fig.suptitle(f'{STEM}: Karnopp truth vs tanh friction cc tanh(g v)', y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(OUT, f'tanh_sweep_{STEM}.png')
    fig.savefig(p, dpi=150)
    print('saved', p)


if __name__ == '__main__':
    main()
