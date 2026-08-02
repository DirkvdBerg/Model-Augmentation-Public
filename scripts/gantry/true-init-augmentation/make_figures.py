"""One summary figure for the true-init result.

Three panels, in the order the argument runs:

  (a) per-window DC scatter per state, three seeding arms against the free-run
      floor. Grouped bars on a log axis, because the quantities span six decades
      and the whole point is which arm sits at the floor and which does not.
  (b) the mechanism: per-window Y mean against the truth's vdelta_a at the window
      start, with the closed-form slope drawn as a line rather than fitted.
  (c) where the absorber state is observable from: instantaneous features
      (what the ANN reads) against the encoder's 18-sample window, per record.

Palette: three categorical hues validated with the dataviz skill's checker
(#0072B2, #D55E00, #009E73; all six checks PASS at the light surface, worst
adjacent CVD dE 11.0). Legend present on every panel with more than one series,
direct labels where they fit, recessive grid, one axis per panel.

Run:
  ... python -u scripts/gantry/true-init-augmentation/make_figures.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from gantry_dynamic.oracle import MA                              # noqa: E402
from model_augmentation.systems.gantry_ss import mh as _mh        # noqa: E402

DIAG = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
FIG = os.path.join(HERE, 'figures')
REC = 'V1_standstill_Yp10'
STATES = ['X', r'$\Theta$', 'Y', r'$\dot X$', r'$\dot\Theta$', r'$\dot Y$']
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
INK, MUTED, GRID = '#1a1a1a', '#5c5c5c', '#d9d9d9'


def style(ax):
    ax.set_facecolor('#fcfcfb')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.8)
    ax.grid(True, which='major', color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def main():
    W = json.load(open(os.path.join(DIAG, 'true_init_window_target.json')))
    O = json.load(open(os.path.join(DIAG, 'true_init_absorber_observability.json')))
    z = np.load(os.path.join(FIG, f'_winmeans_{REC}.npz'))

    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.24)

    # ── (a) per-window DC scatter ────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    d = W['records'][REC]['f64_cog_on']
    arms = (('record seed (finite-difference velocities)', 'record', BLUE),
            ('exact seed (analytic velocities, this experiment)', 'exact', ORANGE),
            ('free run from the exact rest IC (the floor)', 'freerun', GREEN))
    x = np.arange(6)
    for i, (lab, k, c) in enumerate(arms):
        ax.bar(x + (i - 1) * 0.26, d[k]['scatter'], width=0.24, color=c,
               edgecolor='#fcfcfb', linewidth=1.2, label=lab, zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n[{u}]' for s, u in
                        zip(STATES, ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s'])])
    ax.set_ylabel('per-window DC scatter', color=INK, fontsize=9)
    ax.set_title('(a)  Seeding the six physical states EXACTLY does not clean the target\n'
                 r'      bar labels are record/exact: $1.0\times$ means the exact '
                 'velocities bought nothing',
                 loc='left', color=INK, fontsize=11, pad=10)
    for c in range(6):
        r, e = d['record']['scatter'][c], d['exact']['scatter'][c]
        ax.annotate(f'{r/max(e,1e-30):.1f}x', (c, max(r, e) * 1.7), ha='center',
                    fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', ncol=1,
              labelcolor=INK)
    ax.set_ylim(1e-9, 3e-1)

    # ── (b) the mechanism ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    import importlib
    de = importlib.import_module('data_exact')
    tr = de.exact_truth(REC)
    vda = tr['x8'][:, 7][z['starts']]
    ax.scatter(vda * 1e3, z['exact'][:, 2] * 1e6, s=9, color=ORANGE, alpha=0.65,
               linewidths=0, zorder=3, label='per-window mean, exact 6-state seed')
    g = np.linspace(vda.min(), vda.max(), 2)
    slope = -(float(MA) / float(_mh)) * 400 * (1 / 4000) / 2
    ax.plot(g * 1e3, slope * g * 1e6, color=INK, linewidth=1.6, zorder=4,
            label=r'$-\,(m_a/m_h)\,\dot\delta_a(s)\,n_f T_s/2$  (not fitted)')
    ax.set_xlabel(r'truth $\dot\delta_a$ at the window start  [mm/s]', color=INK, fontsize=9)
    ax.set_ylabel(r'per-window mean Y error  [$\mu$m]', color=INK, fontsize=9)
    ax.set_title('(b)  It is the absorber initial condition, $R^2 = 1.0000$',
                 loc='left', color=INK, fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right', labelcolor=INK)

    # ── (c) observability ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style(ax)
    recs = list(O['records'].keys())
    xi = np.arange(len(recs))
    inst = [O['records'][r]['vdelta_a']['inst_test'] for r in recs]
    win = [O['records'][r]['vdelta_a']['win_test'] for r in recs]
    ax.bar(xi - 0.19, inst, width=0.36, color=ORANGE, edgecolor='#fcfcfb',
           linewidth=1.2, zorder=3, label=r'$[x_{\rm phys}(k),\,u(k)]$  (what the ANN reads)')
    ax.bar(xi + 0.19, win, width=0.36, color=BLUE, edgecolor='#fcfcfb',
           linewidth=1.2, zorder=3, label=r'$y,u$ over $k-17\ldots k$  (what the encoder reads)')
    for i, v in enumerate(inst):
        ax.annotate(f'{v:.2f}', (i - 0.19, v + 0.03), ha='center', fontsize=8, color=MUTED)
    ax.set_xticks(xi)
    ax.set_xticklabels([r.split('_')[0] + '\n' + r.split('_')[1] for r in recs],
                       fontsize=8)
    ax.set_ylim(0, 1.52)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r'held-out $R^2$ of $\dot\delta_a$', color=INK, fontsize=9)
    ax.set_title('(c)  The information is in the window, not in the sample',
                 loc='left', color=INK, fontsize=11, pad=10)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left', labelcolor=INK)

    fig.patch.set_facecolor('#fcfcfb')
    out = os.path.join(FIG, 'true_init_summary.png')
    fig.savefig(out, dpi=170, bbox_inches='tight', facecolor='#fcfcfb')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
