"""The data figures of the control-group talk, drawn from caches of earlier sessions only.

    python data_figures.py            # all figures, a few seconds
    python data_figures.py trace      # one of: trace blocksize obc overlap params

Nothing is simulated here. Sources (handoff 2026-09-29, section 4):
  trace      meeting-07-09-2026/code/figure_data_81262.npz: closed-loop re-simulation of the
             four held-out records V1 to V4, controller in the loop, 8-state run 81262.
  blocksize  server/8126x/gantry_results_8126x.npz: `loss_val` (validation closed-loop
             sim-RMS [m]) against `epoch_id`. Only the histories are used from these files:
             their `y_hat_*` traces are OPEN-LOOP free runs and diverge.
  obc        meeting-18-09-2026/figures/OBC/obc_data.json, runs 84032 (no projection) and
             84033 (projection): fit `val` [m] and combined parameter error `combo_err` [%].
  overlap    meeting-21-09-2026/figures/orthogonality_data.json, keys noproj/tangent, span J,
             field `applied`: rho = ||A A^+ F|| / ||F||.
  params     obc_data.json per-parameter errors [%], all ten combinations, backup slide.

Every printed number is checked against the handoff's section 4 values by `check()`.
"""
import json
import sys

import numpy as np

from style import (AUG, BASE, INK, MUTED, OBC_JSON, ORTH_JSON, PROJ, SERVER, SKY,
                   TRACE_CACHE, UM, save, sci, sci_axis, style)

TRACE_RUN = '81262'
# V4 (Lissajous, Y = -0.10 m): its baseline/augmented RMS ratio over the three measured positions
# (3.71) is the closest of the four records to the 3.57 aggregate, so it is the typical record.
TRACE_REC, TRACE_TAG = 3, 'V4_lissajous_Ym10'
Y_CH = 2                       # measured outputs X1, X2, Y; the absorber acts along Y
FS = 4000.0
# HEURISTIC: crop the encoder start transient, the first 0.05 s of the closed-loop run.
CROP_S = 0.05
ZOOM_S = 0.04                  # about 8 periods of the 212 Hz mode
UPDATES_PER_EPOCH = 130        # obc runs: 150 epochs x 130 updates = 19500


def check(label, value, expected, rtol=0.005):
    ok = abs(value - expected) <= rtol * abs(expected)
    print('  [%s] %-44s %.6g (expected %.6g)' % ('ok' if ok else 'MISMATCH', label, value,
                                                  expected))
    if not ok:
        raise SystemExit('number does not reproduce the handoff value: %s' % label)


# ================================================================================================
def fig_trace():
    """Payload-axis error, physics model vs augmented, on one held-out record, closed loop."""
    d = np.load(str(TRACE_CACHE) % TRACE_RUN, allow_pickle=True)
    check('aggregate physics model [um]', float(d['baseline_agg']) * UM, 20.62)
    check('aggregate augmented [um]', float(d['full_agg']) * UM, 5.77)
    k0 = int(d['k0'])
    y = d['y_data_%d' % TRACE_REC][:, Y_CH].astype(float)
    e_base = (y - d['baseline_y_hat_%d' % TRACE_REC][:, Y_CH]) * UM
    e_aug = (y - d['full_y_hat_%d' % TRACE_REC][:, Y_CH]) * UM
    t = (np.arange(len(y)) + k0) / FS
    keep = t >= t[0] + CROP_S
    t, e_base, e_aug = t[keep], e_base[keep], e_aug[keep]
    rms_b, rms_a = np.sqrt(np.mean(e_base ** 2)), np.sqrt(np.mean(e_aug ** 2))
    print('  record %s, Y: physics %.2f um, augmented %.2f um RMS (ratio %.2f)'
          % (TRACE_TAG, rms_b, rms_a, rms_b / rms_a))
    # dominant frequency of each error, to state on the slide what the zoom shows
    for name, e in (('physics', e_base), ('augmented', e_aug)):
        f = np.fft.rfftfreq(len(e), 1 / FS)
        p = np.abs(np.fft.rfft(e - e.mean())) ** 2
        print('  dominant frequency of the %s error: %.1f Hz' % (name, f[np.argmax(p)]))
    # zoom: the window whose physics-model RMS is closest to the record RMS (typical, not best)
    n = int(round(ZOOM_S * FS))
    starts = np.arange(0, len(t) - n, 20)
    loc = np.array([np.sqrt(np.mean(e_base[s:s + n] ** 2)) for s in starts])
    s0 = int(starts[np.argmin(np.abs(loc - rms_b))])
    zw = slice(s0, s0 + n)

    plt = style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.4),
                                   gridspec_kw=dict(width_ratios=[2.0, 1.0], wspace=0.22))
    for ax, sl in ((axL, slice(None)), (axR, zw)):
        ax.plot(t[sl], e_base[sl] / UM, color=BASE, lw=1.0 if ax is axL else 2.0, zorder=2)
        ax.plot(t[sl], e_aug[sl] / UM, color=AUG, lw=1.0 if ax is axL else 2.0, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    lim = 1.08 * np.abs(e_base).max() / UM
    for ax in (axL, axR):
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('time [s]')
    axL.set_xlim(t[0], t[-1])
    axR.set_xlim(t[zw][0], t[zw][-1])
    axR.set_title('zoom, %.0f ms' % (1e3 * ZOOM_S), fontsize=14, color=MUTED, pad=6)
    axR.tick_params(labelleft=False)
    axL.set_ylabel('payload position error [m]')
    sci_axis(axL)
    # the zoom window marked on the full record
    axL.axvspan(t[zw][0], t[zw][-1], color=MUTED, alpha=0.18, lw=0, zorder=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BASE, lw=3),
               Line2D([], [], color=AUG, lw=3)]
    axL.legend(handles, ['physics model, %s m RMS' % sci(rms_b / UM),
                         'augmented model, %s m RMS' % sci(rms_a / UM)],
               loc='lower left', bbox_to_anchor=(0.12, 1.06), ncol=2, handlelength=1.4,
               columnspacing=1.6, borderaxespad=0)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.16)
    save(fig, 'closedloop-trace.png')
    plt.close(fig)


# ================================================================================================
def fig_blocksize():
    """Validation closed-loop error against epoch, 8-state and 2-state learned block."""
    runs = [('81262', AUG, '-', '8 learned states, 10 871 weights', 5.77),
            ('81265', SKY, (0, (5, 2.5)), '2 learned states, 4 019 weights', 5.95)]
    plt = style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    base = None
    for run, c, ls, label, expect in runs:
        r = np.load(SERVER / run / ('gantry_results_%s.npz' % run), allow_pickle=True)
        ep = np.asarray(r['epoch_id'], float)
        lv = np.asarray(r['loss_val'], float) * UM
        check('%s final validation [um]' % run, lv[-1], expect, rtol=0.002)
        base = lv[0] if base is None else base
        ax.plot(ep, lv / UM, color=c, ls=ls, lw=2.6, label=label, zorder=3)
        ax.annotate('%s m' % sci(lv[-1] / UM), xy=(ep[-1], lv[-1] / UM),
                    xytext=(6, 9 if run == '81265' else -11), textcoords='offset points',
                    ha='left', va='center', fontsize=14, color=INK)
    check('epoch-0 validation = physics model [um]', base, 20.62)
    ax.axhline(base / UM, color=BASE, lw=1.6, ls=(0, (2, 2)), zorder=2)
    ax.annotate('physics model, %s m' % sci(base / UM), xy=(350, base / UM), xytext=(0, 6),
                textcoords='offset points', ha='right', va='bottom', fontsize=14, color=INK)
    ax.set_xlim(0, 350)
    ax.set_ylim(0, base / UM * 1.15)
    ax.set_xlabel('training epoch')
    ax.set_ylabel('closed-loop error, held-out [m]')
    sci_axis(ax)
    ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.55), handlelength=2.4)
    fig.subplots_adjust(left=0.12, right=0.80, top=0.93, bottom=0.15)
    save(fig, 'blocksize-curves.png')
    plt.close(fig)


# ================================================================================================
def _obc():
    with open(OBC_JSON, encoding='utf-8') as fh:
        data = json.load(fh)
    arms = {a['run']: a for a in data['arms']}
    return data, arms['84032'], arms['84033']


def _series(arm, key):
    x = [r['it'] / UPDATES_PER_EPOCH for r in arm['records'] if r.get(key) is not None]
    y = [r[key] for r in arm['records'] if r.get(key) is not None]
    return np.asarray(x), np.asarray(y, float)


OBC_ARMS = [('84032', AUG, '-', 'o', 'no projection'),
            ('84033', PROJ, (0, (5, 2)), 's', 'with projection')]


def fig_obc():
    """Fit and combined parameter error against epoch, joint estimation from a 10 % detuned start."""
    data, a32, a33 = _obc()
    arms = {'84032': a32, '84033': a33}
    check('fit at epoch 0 [um]', a32['records'][0]['val'] * UM, 22.97)
    check('fit final, no projection [um]', a32['records'][-1]['val'] * UM, 5.86)
    check('fit final, projection [um]', a33['records'][-1]['val'] * UM, 5.89)
    check('combined parameter error at start [%]', a32['combo_start'], 9.49)
    check('combined parameter error final, no projection [%]', a32['combo_final'], 7.20)
    check('combined parameter error final, projection [%]', a33['combo_final'], 7.69)
    check('projection minimum [%]', a33['combo_min'], 6.98)
    # combined error = RMS over the ten per-parameter errors (reproduces the logged value)
    p = a32['records'][-1]['params']
    check('RMS of the ten final errors, no projection', np.sqrt(np.mean(
        [v ** 2 for v in p.values()])), 7.20, rtol=0.01)

    plt = style()
    fig, (axF, axP) = plt.subplots(1, 2, figsize=(12.0, 4.5),
                                   gridspec_kw=dict(wspace=0.62))
    for run, c, ls, mk, label in OBC_ARMS:
        x, v = _series(arms[run], 'val')
        axF.plot(x, v, color=c, ls=ls, marker=mk, ms=7, mfc='white', mew=1.8,
                 label=label, zorder=3)
        x, ce = _series(arms[run], 'combo_err')
        axP.plot(x, ce, color=c, ls=ls, marker=mk, ms=7, mfc='white', mew=1.8, zorder=3)
    # end values, in ink, beside the line ends
    for run, dy_f, dy_p in (('84032', -12, -12), ('84033', 12, 12)):
        _, v = _series(arms[run], 'val')
        _, ce = _series(arms[run], 'combo_err')
        axF.annotate('%s m' % sci(v[-1]), xy=(150, v[-1]), xytext=(8, dy_f),
                     textcoords='offset points', ha='left', va='center', fontsize=14)
        axP.annotate('%.2f %%' % ce[-1], xy=(150, ce[-1]), xytext=(8, dy_p),
                     textcoords='offset points', ha='left', va='center', fontsize=14)
    v0 = a32['records'][0]['val']
    axF.annotate('start: physics model,\nparameters 10 %% off, %s m' % sci(v0),
                 xy=(0, v0), xytext=(14, -4), textcoords='offset points', ha='left',
                 va='top', fontsize=14)
    s0 = a32['combo_start']
    axP.axhline(s0, color=MUTED, lw=1.2, ls=(0, (2, 2)), zorder=1)
    axP.annotate('start, %.1f %%' % s0, xy=(150, s0), xytext=(0, 5),
                 textcoords='offset points', ha='right', va='bottom', fontsize=14)
    axF.set_ylim(0, 2.5e-5)
    axP.set_ylim(6.0, 10.0)
    for ax in (axF, axP):
        ax.set_xlim(0, 150)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xlabel('training epoch')
    axF.set_ylabel('closed-loop error [m]')
    sci_axis(axF)
    axP.set_ylabel('parameter error [%]')
    axF.legend(loc='center right', bbox_to_anchor=(1.0, 0.55), handlelength=2.6)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.92, bottom=0.16)
    save(fig, 'obc-fit-params.png')
    plt.close(fig)


# ================================================================================================
def fig_overlap():
    """Share of the learned correction inside the span of the ten parameter directions."""
    with open(ORTH_JSON, encoding='utf-8') as fh:
        o = json.load(fh)
    vals = {(arm, split): o[arm][split]['J']['applied']
            for arm in ('noproj', 'tangent') for split in ('train', 'val')}
    check('overlap no projection, train', vals['noproj', 'train'], 0.865, rtol=0.002)
    check('overlap no projection, held-out', vals['noproj', 'val'], 0.870, rtol=0.002)
    check('overlap projection, train', vals['tangent', 'train'], 8.3e-8, rtol=0.01)
    check('overlap projection, held-out', vals['tangent', 'val'], 0.334, rtol=0.002)

    plt = style()
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    w, gap = 0.34, 0.03
    groups = [('train', 'training records'), ('val', 'held-out records')]
    for g, (split, gname) in enumerate(groups):
        for j, (arm, c, label) in enumerate((('noproj', AUG, 'no projection'),
                                             ('tangent', PROJ, 'with projection'))):
            x = g + (j - 0.5) * (w + gap)
            v = vals[arm, split]
            ax.bar(x, v, width=w, color=c, label=label if g == 0 else None, zorder=3)
            txt = '%.2f' % v if v > 1e-3 else sci(v, 1)
            ax.annotate(txt, xy=(x, v), xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=15, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[1] for g in groups], fontsize=16, color=INK)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel(r'overlap $\rho$ with parameter directions')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2,
              columnspacing=1.4)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.11)
    save(fig, 'overlap-bars.png')
    plt.close(fig)


# ================================================================================================
PARAM_ROWS = [['mh', 'm_total', 'm_diff', 'J_eff', 'd'],
              ['kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum']]
PARAM_NAME = {
    'mh': 'payload mass', 'm_total': 'total mass', 'm_diff': 'drive mass difference',
    'J_eff': 'yaw inertia', 'd': 'payload offset d', 'kb_sum': 'joint stiffness',
    'cg1': 'drive 1 damping', 'cg2': 'drive 2 damping', 'cy': 'payload drive damping',
    'cb_sum': 'joint damping',
}


def fig_params():
    """Backup: error of each of the ten parameter combinations, both arms."""
    data, a32, a33 = _obc()
    arms = {'84032': a32, '84033': a33}
    assert sorted(PARAM_NAME) == sorted(data['param_order']), 'all ten parameters, no subset'
    check('no projection cg1 final [%]', a32['records'][-1]['params']['cg1'], 16.99, rtol=0.002)
    check('projection kb_sum final [%]', a33['records'][-1]['params']['kb_sum'], 15.52, rtol=0.002)
    check('projection d final [%]', a33['records'][-1]['params']['d'], 13.9, rtol=0.002)

    plt = style()
    fig, axes = plt.subplots(2, 5, figsize=(12.4, 5.6), sharex=True, sharey=True)
    for r, row in enumerate(PARAM_ROWS):
        for c, name in enumerate(row):
            ax = axes[r][c]
            ax.axhline(0, color=INK, lw=1.0, zorder=1)
            for run, col, ls, mk, label in OBC_ARMS:
                x = [q['it'] / UPDATES_PER_EPOCH for q in arms[run]['records']
                     if name in q.get('params', {})]
                y = [q['params'][name] for q in arms[run]['records']
                     if name in q.get('params', {})]
                ax.plot(x, y, color=col, ls=ls, lw=2.0, zorder=3,
                        label=label if (r, c) == (0, 0) else None)
            ax.set_title(PARAM_NAME[name], fontsize=14, loc='left', pad=4)
            ax.set_xlim(0, 150)
            ax.set_xticks([0, 75, 150])
            ax.set_ylim(-22, 22)
            ax.set_yticks([-20, -10, 0, 10, 20])
            ax.tick_params(labelsize=13)
            if c == 0:
                ax.set_ylabel('error [%]', fontsize=14)
            if r == 1:
                ax.set_xlabel('epoch', fontsize=14)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0),
               handlelength=2.6)
    fig.subplots_adjust(left=0.07, right=0.975, top=0.94, bottom=0.2, hspace=0.35, wspace=0.24)
    save(fig, 'params-all.png')
    plt.close(fig)


# ================================================================================================
def fig_frf():
    """F_Y -> Y frequency response: physics model against the simulated machine, Y = 0.

    Both from their linear equations of motion at Y = 0 (delta_a = 0), open-loop plant, not from
    the closed-loop data. Parameters: Matlab-scripts/Augmentation/data/gtd_config.m (physical
    block, fa = 150 Hz) with the dataset overrides of generate_trajectory_data.m
    (augmentation_ma50_b140-230_a6_z03: MA_FRAC 0.50, ZETA_A_OVERRIDE 0.03). Truth mass matrix:
    Matlab-scripts/Augmentation/gantrySystemExtended.m.
    """
    m1, m2, mb, mh_tot, Lb, Jb, Jh, d = 10.2, 10.7, 22.8, 10.1, 0.725, 1.0, 0.05, 0.1
    cg1, cg2, cy, cb, kb = 14.5, 20.3, 10.0, 9.0 + 9.0, 1987.5 + 1987.5
    ma = 0.5 * mh_tot
    mh = mh_tot - ma
    fa, zeta_a, L0 = 150.0, 0.03, 0.10
    ka = ma * (2 * np.pi * fa) ** 2              # THEORY: k = m (2 pi f)^2, gtd_config.m
    ca = 2 * zeta_a * np.sqrt(ka * ma)           # THEORY: c = 2 zeta sqrt(k m), gtd_config.m
    C3 = np.array([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0],
                   [(cg1 - cg2) * Lb / 2, cb + (cg1 + cg2) * Lb ** 2 / 4, 0],
                   [0, 0, cy]])
    K3 = np.diag([0, kb, 0])
    J0 = Jb + Jh + (m1 + m2) * Lb ** 2 / 4
    Mb = np.array([[m1 + m2 + mb + mh_tot, (m1 - m2) * Lb / 2, 0],
                   [(m1 - m2) * Lb / 2, J0 + mh_tot * d ** 2, -mh_tot * d],
                   [0, -mh_tot * d, mh_tot]])            # thesis Eq. (mass_matrix) at Y = 0
    Mt = np.array([[m1 + m2 + mb + mh_tot, (m1 - m2) * Lb / 2 - ma * L0, 0, 0],
                   [(m1 - m2) * Lb / 2 - ma * L0, J0 + mh_tot * d ** 2 + ma * L0 ** 2,
                    -mh_tot * d, -ma * d],
                   [0, -mh_tot * d, mh_tot, ma],
                   [0, -ma * d, ma, ma]])               # gantrySystemExtended.m, Y = delta_a = 0
    Ct = np.zeros((4, 4)); Ct[:3, :3] = C3; Ct[3, 3] = ca
    Kt = np.zeros((4, 4)); Kt[:3, :3] = K3; Kt[3, 3] = ka
    f = np.logspace(np.log10(20), 3, 4000)

    def frf(M, C, K):
        n = M.shape[0]
        b = np.zeros(n); b[2] = 1.0                      # force F_Y on the Y coordinate
        return np.array([abs(np.linalg.solve(-(2 * np.pi * fi) ** 2 * M
                                             + 2j * np.pi * fi * C + K, b)[2]) for fi in f])

    Gb, Gt = frf(Mb, C3, K3), frf(Mt, Ct, Kt)
    band = (f > 120) & (f < 260)
    f_dip = f[band][np.argmin(Gt[band] / Gb[band])]
    f_peak = f[band][np.argmax(Gt[band] / Gb[band])]
    check('anti-resonance [Hz]', f_dip, 150.0, rtol=0.02)
    check('resonance [Hz]', f_peak, 212.13, rtol=0.02)

    plt = style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axvspan(140, 230, color=MUTED, alpha=0.12, lw=0, zorder=0)
    ax.loglog(f, Gb, color=BASE, lw=2.4, label='physics model', zorder=3)
    ax.loglog(f, Gt, color=INK, lw=2.4, label='simulated system', zorder=4)
    ax.set_xlim(20, 1000)
    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.set_xticklabels(['20', '50', '100', '200', '500', '1000'])
    ax.minorticks_off()
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel(r'$|Y/F_Y|$ [m/N]')
    ax.grid(which='minor', visible=False)
    ax.text(185, ax.get_ylim()[0] * 1.6, 'excitation', ha='center', va='bottom',
            fontsize=13, color=MUTED)
    ax.legend(loc='upper right')
    fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.15)
    save(fig, 'frf-physics-vs-machine.png')
    plt.close(fig)


def fig_inner_build():
    """Two-step build of the inner-product figure, without the 'network output before removal'
    bars: version 1 only no projection, version 2 adds the correction that reaches the model.
    Bar slots are fixed across both, so version 2 is version 1 with the green bars added."""
    fig_inner(bars_shown=('noproj',), slots=('noproj', 'tangent'),
              name='inner-products-step1-noproj.png', note=False)
    fig_inner(bars_shown=('noproj', 'tangent'), slots=('noproj', 'tangent'),
              name='inner-products-step2-projection.png', note=True)


def fig_inner(bars_shown=('noproj', 'raw', 'tangent'), slots=('noproj', 'raw', 'tangent'),
              name='inner-products.png', note=True):
    """Normalised inner product with each of the ten parameter directions, both runs.

    Reads inner_products.json (collect_inner_products.py, run against the model code from before
    D-229 because the current parameter coordinate cannot load 84032/84033). Ten directions only,
    no affine offset.
    """
    import pathlib
    d = json.loads((pathlib.Path(__file__).resolve().parent / 'inner_products.json')
                   .read_text(encoding='utf-8'))
    check('rho no projection, train', d['noproj']['train']['applied']['rho'], 0.865, rtol=0.002)
    check('rho projection applied, train', d['tangent']['train']['applied']['rho'], 8.3e-8,
          rtol=0.01)
    check('rho projection applied, held-out', d['tangent']['val']['applied']['rho'], 0.334,
          rtol=0.002)
    names = ['kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum', 'mh', 'm_total', 'm_diff', 'J_eff', 'd']
    labels = ['joint\nstiffness', 'drive 1\ndamping', 'drive 2\ndamping', 'payload\ndrive damp.',
              'joint\ndamping', 'payload\nmass', 'total\nmass', 'drive mass\ndifference',
              'yaw\ninertia', 'payload\noffset d']
    bars = {'noproj': ('noproj', 'applied', AUG, None, 'no projection: network correction'),
            'raw': ('tangent', 'raw', 'white', PROJ,
                    'with projection: network output before removal'),
            'tangent': ('tangent', 'applied', PROJ, None,
                        'with projection: correction that reaches the model')}
    plt = style()
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 6.6), sharex=True)
    x = np.arange(10)
    w = 0.27
    for ax, split, title in zip(axes, ('train', 'val'),
                                ('training records', 'held-out records')):
        ax.axhline(0, color=INK, lw=1.0)
        for key in bars_shown:
            arm, kind, face, edge, label = bars[key]
            i = slots.index(key)
            v = [d[arm][split][kind]['cos'][n] for n in names]
            ax.bar(x + (i - (len(slots) - 1) / 2) * w, v, w, color=face, edgecolor=edge or face,
                   hatch='///' if edge else None, lw=1.2, label=label, zorder=3)
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlim(-0.6, 9.6)
        ax.set_ylabel('normalised\ninner product')
        ax.set_title(title, loc='left', fontsize=15, pad=4)
        ax.grid(axis='x', visible=False)
    mx = max(abs(v) for v in d['tangent']['train']['applied']['cos'].values())
    if note:
        axes[0].annotate('correction that reaches the model: every bar below %s' % sci(mx, 0),
                         xy=(0.99, 0.06), xycoords='axes fraction', ha='right', va='bottom',
                         fontsize=13, color=INK)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=12)
    axes[1].tick_params(axis='x', length=0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower left', ncol=3, fontsize=12, bbox_to_anchor=(0.1, 0.0),
               handlelength=1.6, columnspacing=1.2)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.2, hspace=0.25)
    save(fig, name)
    plt.close(fig)


FIGURES = {'frf': fig_frf, 'inner': fig_inner, 'inner_build': fig_inner_build, 'trace': fig_trace, 'blocksize': fig_blocksize, 'obc': fig_obc,
           'overlap': fig_overlap, 'params': fig_params}

if __name__ == '__main__':
    for key in (sys.argv[1:] or list(FIGURES)):
        print('== %s' % key)
        FIGURES[key]()
