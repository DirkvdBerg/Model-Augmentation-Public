"""The five settling figures. Redraws from `settling_data.npz`; never simulates.

    python fig_settling.py

Layout is the 2026-09-07 trajectory figure's, deliberately: three rows X1 / X2 / Y, the FULL
record on the left, a zoom on the right, separate y-scales per column, every panel naming the
record and the loop mode. The one change is WHAT the right column zooms on. There it was the most
stationary window, because the figure was about a multisine riding on a dwell. Here it is a
SETTLING window, from the end of the largest move through the dwell that follows it, because that
is what Quinten asked to see.

Output:
    settling_<arm>.png/pdf   one per arm: system vs that model alone, four files
    settling_all.png/pdf     the data plus all four models overlaid

No settling time is quoted and no tolerance band is drawn, per Quinten: "Voor het model vooral
het gedrag van hoe die settled, system afhankelijk niet het model". The settling is the plant's
property; what these figures compare is how each model reproduces it.

The same window is used in every one of the five files, so they can be laid side by side.
"""
__project_origin__ = "added"

import sys

import numpy as np

from fig_common import (ARMS, ARM_ORDER, CHANNELS, C_DATA, FIGDIR, FS, MUTED, RECORD, finish_columns,
                    load_cache, pick_window, sci_axis, settling_windows, style, two_col_axes)


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        p = FIGDIR / ('%s.%s' % (name, ext))
        fig.savefig(p)
    print('  wrote %s.{png,pdf}' % (FIGDIR / name))


def _panels(plt, d, arms, title, name, mode='abs'):
    """One 3x2 figure: full record left, settling zoom right, for the arms named.

    `mode='err'` draws BOTH columns as MODEL ERROR, `y_hat - y_data`, one trace per arm and no
    system trace: the full record on the left, the settling zoom on the right. All three
    channels, not Y alone.

    Two subtractions were possible here and only one of them says anything. `y - r` removes the
    offset notation matplotlib is forced into by absolute position (`x10^-6` sitting on top of
    `-1.5513x10^-1`), but `r` is CONSTANT inside a dwell, so subtracting it is a pure axis shift:
    identical curves, relabelled axis, no new information. `y_hat - y_data` is a different
    quantity. It is what the rms table reports, it lives at 1e-07 to 1e-06 m, and it is
    invisible in the absolute view because it sits under traces drawn at 3e-05 m. On its own
    axis it is the panel that separates the arms.

    Keeping the full record in error form as well shows whether the chosen settling window is
    representative or unusual, which the absolute view cannot answer.
    """
    y, r, k0 = d['y_true'], d['r'], int(d['k0'])
    t = (np.arange(len(y)) + k0) / FS
    i0, i1 = pick_window(r)
    twin = (float(t[i0]), float(t[i1 - 1]))
    err = (mode == 'err')

    fig, axes = two_col_axes(plt, 3, figsize=(9.2, 5.4))
    for row, ch in enumerate(CHANNELS):
        for col, sl in ((0, slice(None)), (1, slice(i0, i1))):
            ax = axes[row][col]
            if err:
                ax.axhline(0.0, color=MUTED, lw=0.7, ls=':')
            else:
                ax.plot(t[sl], y[sl, row], color=C_DATA, lw=1.4, label='system', zorder=10)
            for a in arms:
                s = ARMS[a]
                v = d['yhat_' + a][sl, row] - (y[sl, row] if err else 0.0)
                ax.plot(t[sl], v, color=s['c'], ls=s['ls'], lw=s['lw'],
                        label=s['label'], zorder=s['z'])
            sci_axis(ax)
        axes[row][0].set_ylabel('%s error [m]' % ch if err else '%s [m]' % ch)

    n_win = len(settling_windows(r))
    finish_columns(axes, t, twin, zoom_note='settling, largest move')
    axes[0][0].legend(loc='upper left', ncol=2, frameon=False, fontsize=7)
    fig.suptitle('%s   closed loop   %s\n%s%d settling windows in this record, '
                 'no multisine (f_sim = 0)'
                 % (RECORD, title, 'model error $\\hat y - y$   ' if err else '', n_win),
                 fontsize=9.5, y=1.015)
    fig.tight_layout()
    _save(fig, name)
    plt.close(fig)
    return twin, n_win


def main():
    plt = style()
    d = load_cache()

    print('per-arm NRMS on %s  (rms error / ystd, per channel)' % RECORD)
    for a in ARM_ORDER:
        print('  %-9s %s   [V2 check: %s]'
              % (a,
                 np.array2string(d['nrms_' + a], formatter={'float_kind': lambda v: '%9.5f' % v}),
                 np.array2string(d['nrms_v2_' + a],
                                 formatter={'float_kind': lambda v: '%9.5f' % v})))

    print('\ndrawing')
    for a in ARM_ORDER:
        _panels(plt, d, [a], ARMS[a]['label'], 'settling_%s' % a)
    twin, n_win = _panels(plt, d, ARM_ORDER, 'all arms', 'settling_all')

    # Same five figures with the zoom column as error from the dwell setpoint.
    for a in ARM_ORDER:
        _panels(plt, d, [a], ARMS[a]['label'], 'settling_err_%s' % a, mode='err')
    _panels(plt, d, ARM_ORDER, 'all arms', 'settling_err_all', mode='err')
    print('\nzoom window %.3f to %.3f s, chosen as the dwell after the largest move; '
          '%d settling windows available' % (twin[0], twin[1], n_win))


if __name__ == '__main__':
    sys.exit(main())
