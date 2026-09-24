"""The four presentation figures of the encoder-isolation arm (D-197).

One figure per question the supervisors asked, and one they did not:

  A  encoder-init-vs-truth       "encoder init compare to actual states"
  B  encoder-supervised-limit    "only train the encoder ... verify the structure is good enough"
  C  encoder-window-sweep        "encoder window"
  D  freerun-by-initial-state    what the state error COSTS, and why a better encoder can hurt

HOUSE STYLE (`docs/writeup/figure-style.md`): black on white, greys carry meaning and colour does
not, direct labelling in preference to a legend, units in brackets, no em-dashes. The greys are
the same three tokens the TikZ figures use: black for the object under discussion, `0.45` for
something present but explicitly not the claim, `0.88` for a band. Series are separated by dash
pattern, never by hue, so these survive a photocopier and a projector like the block schemes do.
Native width is the project default of 175 mm, so they may be placed down to 0.85x (149 mm).

Everything is drawn from cached results; nothing is re-simulated and nothing is re-trained.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u scripts/gantry/transient/code/make_figures.py
"""
__project_origin__ = "added"

import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import enc_common as ec                                          # noqa: E402

BLACK, MID, LIGHT = 'black', '0.45', '0.88'
MM = 1.0 / 25.4
W_PAGE = 175 * MM                       # native width; placed at >= 0.85x per the style guide
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
# The four presentation figures land in the MEETING folder, not in the arm's own `figure/`, which
# holds the per-run diagnostic plots the experiment scripts write. Keeping them apart stops a
# rerun of this script from quietly recreating a second copy next to the one being presented.
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'meeting',
                   'meeting-18-09-2026', 'figures', 'transient')

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.grid': True, 'grid.color': LIGHT, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'axes.edgecolor': BLACK, 'axes.linewidth': 0.8,
    'lines.solid_capstyle': 'round', 'savefig.bbox': 'tight', 'figure.dpi': 150,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
})


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'))
    plt.close(fig)
    print(f'  {name}.pdf / .png')


def fig_a(norm, cfg, record='V1_standstill_Yp10', t0_ms=6000.0, span_ms=40.0):
    """A: truth, stored x_logical and the encoder at init, on Y and dY, with the error below."""
    z = np.load(os.path.join(RES, 'r0_init.npz'), allow_pickle=True)
    k_ix = z[f'{record}|k_ix']
    xt, xfd, xh = z[f'{record}|xt'], z[f'{record}|xfd'], z[f'{record}|x_hat']
    x_mean, std_x = norm.x_mean.flatten(), norm.std_x.flatten()
    t = k_ix * cfg.ts_new * 1e3
    m = (t >= t0_ms) & (t <= t0_ms + span_ms)

    fig, ax = plt.subplots(2, 2, figsize=(W_PAGE, 88 * MM), sharex=True,
                           gridspec_kw={'height_ratios': [2.0, 1.0]})
    for col, (ch, unit) in enumerate(((2, 'm'), (5, 'm/s'))):
        true = xt[m, ch] * std_x[ch] + x_mean[ch]
        stored = xfd[m, ch] * std_x[ch] + x_mean[ch]
        est = xh[m, ch] * std_x[ch] + x_mean[ch]
        off = true.mean()                       # plot about the mean; the offset is not the point
        a = ax[0, col]
        a.plot(t[m], (stored - off) * 1e3, color=MID, lw=2.6, solid_capstyle='butt')
        a.plot(t[m], (true - off) * 1e3, color=BLACK, lw=1.0)
        a.plot(t[m], (est - off) * 1e3, color=BLACK, lw=1.0, ls=(0, (4, 2)))
        lbl = 'Y' if ch == 2 else 'dY'
        a.set_ylabel(f'{lbl} minus its mean\n[{"mm" if ch == 2 else "mm/s"}]')
        a.set_title(lbl, loc='left')

        b = ax[1, col]
        err = (est - true) * 1e3
        b.plot(t[m], err, color=BLACK, lw=0.9)
        b.axhline(0.0, color=MID, lw=0.6)
        rms = np.sqrt((((xh[:, ch] - xt[:, ch]) * std_x[ch]) ** 2).mean())
        sig = (xt[:, ch] * std_x[ch]).std()
        b.set_ylabel(f'encoder error\n[{"mm" if ch == 2 else "mm/s"}]')
        b.set_xlabel('time [ms]')
        b.text(0.015, 0.93,
               f'record RMS {rms:.2e} {unit}  ({100 * rms / sig:.0f}% of the signal std)',
               transform=b.transAxes, fontsize=6.5, va='top',
               bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

    # A legend rather than direct labels: the three traces overlap across the whole window, the
    # case `docs/writeup/figure-style.md` section 7 names as the one a legend is for.
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLACK, lw=1.0, label='exact truth'),
               Line2D([], [], color=BLACK, lw=1.0, ls=(0, (4, 2)),
                      label='encoder at initialisation'),
               Line2D([], [], color=MID, lw=2.6,
                      label='stored x_logical (finite difference), under the truth at this scale')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.005, 1.0), ncol=3,
               frameon=False, handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.945))
    save(fig, 'encoder-init-vs-truth-v1')


def fig_b(tag='r1_exact_long'):
    """B: what direct supervision buys, per channel, relative to the initialisation."""
    z = np.load(os.path.join(RES, f'{tag}.npz'))
    j = json.load(open(os.path.join(RES, f'{tag}.json')))
    init, trained = z['init|rms_phys'], z['trained|rms_phys']
    rel = trained / init
    x = np.arange(len(ec.CHANNELS))

    fig, ax = plt.subplots(figsize=(W_PAGE, 80 * MM))
    ax.bar(x - 0.19, np.ones_like(rel), 0.36, facecolor=LIGHT, edgecolor=BLACK, linewidth=0.8)
    ax.bar(x + 0.19, rel, 0.36, facecolor=BLACK, edgecolor=BLACK, linewidth=0.8)
    for i, r in enumerate(rel):
        ax.text(i + 0.19, r + 0.03, f'{1 / r:.1f}x', ha='center', fontsize=7.5)
    ax.axhline(1.0, color=BLACK, lw=0.8)
    ax.set_ylim(0, 1.30)
    for i, (v, u) in enumerate(zip(init, ec.UNITS)):
        ax.text(i - 0.19, 1.02, f'{v:.1e}\n{u}', ha='center', va='bottom', fontsize=6.5,
                color=MID, linespacing=1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(ec.CHANNELS)
    ax.set_ylabel('state error after training,\nrelative to the initialisation')
    ax.set_title('The encoder trained ALONE against the exact state, no rollout in the loss: '
                 f'{j["args"]["epochs"]} epochs, na = nb = {j["na"]}', loc='left', pad=22)
    # Key above the axes, not inside them: inside, it lay across the bars it describes.
    ax.text(0.0, 1.035,
            'open bar: the reconstructability initialisation, set to 1 per channel, with its '
            'absolute error printed above it.   filled bar: after training.',
            transform=ax.transAxes, fontsize=7, color=MID)
    fig.tight_layout(pad=0.6)
    save(fig, 'encoder-supervised-limit-v1')


def fig_c(cfg):
    """C: error against encoder window length, Y and dY, init and trained at an equal budget."""
    arms = []
    for f in sorted(glob.glob(os.path.join(RES, 'r1_exact_long*.json'))):
        j = json.load(open(f))
        if j['args']['epochs'] != 1000:
            continue
        z = np.load(f.replace('.json', '.npz'))
        arms.append((j['na'], z['init|rms_phys'], z['trained|rms_phys']))
    arms.sort()
    if len(arms) < 3:
        print('  encoder-window-sweep: SKIPPED, fewer than three converged arms on disk')
        return
    na = np.array([a[0] for a in arms])
    ms = (na + 1) * cfg.ts_new * 1e3
    init = np.array([a[1] for a in arms])
    trained = np.array([a[2] for a in arms])

    fig, ax = plt.subplots(1, 2, figsize=(W_PAGE, 70 * MM))
    for col, (ch, lab, unit) in enumerate(((2, 'Y', 'm'), (5, 'dY', 'm/s'))):
        a = ax[col]
        a.axvspan((29 + 1) * cfg.ts_new * 1e3 * 0.93, (29 + 1) * cfg.ts_new * 1e3 * 1.07,
                  color=LIGHT, lw=0)
        a.loglog(ms, init[:, ch], color=MID, lw=1.2, ls=(0, (4, 2)), marker='o', ms=3.5,
                 mfc='white', mec=MID)
        a.loglog(ms, trained[:, ch], color=BLACK, lw=1.2, marker='o', ms=3.5)
        a.set_xlabel('encoder window length [ms]')
        a.set_ylabel(f'{lab} state error, RMS [{unit}]')
        a.set_title(f'{lab}', loc='left')
        a.set_xticks(ms)
        a.set_xticklabels([f'{v:.1f}\nna={n}' for v, n in zip(ms, na)])
        # Kill the x minor ticks only: minorticks_off() also removes the y decade subdivisions,
        # and a log axis showing one labelled decade cannot be read.
        a.xaxis.set_minor_locator(plt.NullLocator())
    # Direct labels anchored to the curves themselves, offset in POINTS so they cannot be pushed
    # outside the axes by the data range the way an offset in data coordinates was.
    box = dict(facecolor='white', edgecolor='none', pad=1.0)
    ax[0].annotate('at initialisation', xy=(ms[1], init[1, 2]), xytext=(0, 9),
                   textcoords='offset points', fontsize=7.5, color=MID, ha='center', bbox=box)
    ax[0].annotate('after training the encoder alone', xy=(ms[2], trained[2, 2]),
                   xytext=(0, -14), textcoords='offset points', fontsize=7.5, ha='center',
                   va='top', bbox=box)
    ax[1].annotate('production setting', xy=(ms[1], trained[1, 5]), xytext=(14, -6),
                   textcoords='offset points', fontsize=7, color=MID, ha='left', bbox=box)
    fig.suptitle('The window is the lever: same 1000 epochs and same rate at every window length',
                 x=0.012, ha='left', fontsize=8)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))
    save(fig, 'encoder-window-sweep-v1')


def fig_d(cfg, H=4000, records=('V1_standstill_Yp10', 'V3_ysweep_Yp10')):
    """D: free-run error on Y from three initial states, untrained model."""
    need = [os.path.join(RES, f'r2fig_{a}.npz') for a in ('init', 'trained')]
    if not all(os.path.exists(f) for f in need):
        print('  freerun-by-initial-state: SKIPPED, run run_figure_inputs.sh first')
        return
    zi = np.load(os.path.join(RES, 'r2fig_init.npz'), allow_pickle=True)
    zt = np.load(os.path.join(RES, 'r2fig_trained.npz'), allow_pickle=True)
    t = np.arange(H) * cfg.ts_new

    fig, ax = plt.subplots(1, len(records), figsize=(W_PAGE, 74 * MM), sharey=True)
    for col, rec in enumerate(records):
        a = ax[col]
        a.axvspan(0, 400 * cfg.ts_new, color=LIGHT, lw=0)
        a.semilogy(t, zi[f'{rec}|{H}|exact|rms_t'][:, 2], color=BLACK, lw=1.1)
        a.semilogy(t, zi[f'{rec}|{H}|encoder|rms_t'][:, 2], color=BLACK, lw=1.1,
                   ls=(0, (4, 2)))
        a.semilogy(t, zt[f'{rec}|{H}|encoder|rms_t'][:, 2], color=MID, lw=1.6)
        a.set_xlabel('time into the free run [s]')
        a.set_title(rec.split('_')[0] + ', ' + ' '.join(rec.split('_')[1:]), loc='left')
        a.text(400 * cfg.ts_new, a.get_ylim()[0], '  training horizon nf = 400',
               fontsize=7, color=MID, va='bottom')
    ax[0].set_ylabel('Y output RMS error over 20 starts [m]')
    # Direct labels in DATA coordinates, anchored to each curve where the three are furthest
    # apart, so no label sits on a trace and no legend is needed.
    rec0, i = records[0], int(0.30 * H)
    for key, z, colour, factor, xf, text in (
            ('encoder', zt, MID, 3.0, 0.16, 'x0 = encoder trained on the truth'),
            ('exact', zi, BLACK, 0.30, 0.30, 'x0 = exact truth'),
            ('encoder', zi, BLACK, 0.26, 0.30, 'x0 = encoder at initialisation')):
        j = int(xf * H)
        y = z[f'{rec0}|{H}|{key}|rms_t'][j, 2]
        ax[0].text(t[j], y * factor, text, fontsize=7.5, color=colour, ha='left',
                   bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
    fig.suptitle('Free run of the UNTRAINED model, open loop: the exact state is the worse seed',
                 x=0.012, ha='left', fontsize=8)
    fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.94))
    save(fig, 'freerun-by-initial-state-v1')


CAPTIONS = {
    'encoder-init-vs-truth-v1':
        'The encoder at its reconstructability initialisation against the exact state, on the two '
        'channels it gets wrong. Record V1_standstill_Yp10, dataset augmentation_ma50_b140-230_'
        'a6_z03, na = nb = 29 (30 samples, 7.5 ms), no training. Top: 40 ms of Y and dY about '
        'their mean. Bottom: encoder minus truth, with the RMS over the whole record. The stored '
        'x_logical is drawn as a wide grey trace and is indistinguishable from the exact state at '
        'this scale: its finite-difference error is a relative 8e-04, three orders below the '
        'encoder error. Real data, not schematic.',
    'encoder-supervised-limit-v1':
        'What direct supervision buys the encoder. The encoder is trained ALONE against the exact '
        'state with a regression loss, baseline and ANN frozen and never simulated, so there is no '
        'rollout between the loss and the encoder. 1000 epochs, Adam at lr 1e-4, na = nb = 29. '
        'Each channel is normalised by its own error at initialisation, printed under the channel '
        'name. Even with the free-run objective removed entirely, the error falls by at most 3x, '
        'and on X and dX not at all. Real data, not schematic.',
    'encoder-window-sweep-v1':
        'The same experiment repeated at four encoder window lengths, at an equal budget of 1000 '
        'epochs and the same rate. Dashed grey with open markers: the reconstructability '
        'initialisation at that window. Black: after training the encoder alone against the exact '
        'state. The band marks the production setting, na = nb = 29. The trained error falls '
        'monotonically with the window on both channels, dY by 8.4x from 4.0 to 30.0 ms, while '
        'the initialisation is flat, so the limit at 7.5 ms is the window rather than the '
        'optimisation. Not shown, and the reason a window change is not free: on the Theta '
        'channels the initialisation gets WORSE as the window grows. Real data, not schematic.',
    'freerun-by-initial-state-v1':
        'What the state error costs in free run, and why a more accurate encoder can hurt. The '
        'model is frozen as built, with the ANN zero-initialised, so the black solid curve is the '
        'baseline mismatch alone. Open loop, recorded plant input replayed, RMS over 20 starts per '
        'record, Y output. Seeding with the encoder at initialisation (dashed) is about ten times '
        'better than seeding with the TRUE state, and seeding with the encoder after it has been '
        'trained toward the truth (grey) gives that advantage back: the reconstructability map is '
        'built from the baseline, so it estimates the state that makes the baseline reproduce the '
        'window, and that bias cancels part of the baseline mismatch. The band is the training '
        'horizon, nf = 400. Real data, not schematic.',
}


def main():
    ec.ensure_dirs()
    os.makedirs(FIG, exist_ok=True)
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    print(f'Figures written to {os.path.relpath(FIG)}:')
    fig_a(norm, cfg)
    fig_b()
    fig_c(cfg)
    fig_d(cfg)
    print('\nCaptions:')
    for k, v in CAPTIONS.items():
        print(f'\n[{k}]\n{v}')


if __name__ == '__main__':
    main()
