"""Live-ResNet seed-1 figures for the 08-10 meeting.

Inputs: resid_{baseline,plain,ps2}.npz from eval_resid (closed-loop free run over the six full validation records,
e = y - y_hat from k0; baseline = the initial model before training, plain = run 88532 best, PS2 = run 88531 best),
and the two Slurm logs (validation sim-RMS every 1300 updates).
Spectrum: Welch ASD, Hann, nperseg 4096 (0.98 Hz), 50 % overlap, fs 4 kHz, PSD averaged over the four multisine
validation records (VA-S1, Y1, P1, L1), never concatenated. Band RMS: one-sided rfft Parseval per record
(band_compare.band_rms), mean over the same four records.
"""
__project_origin__ = "added"

import json, os, re, sys
import numpy as np
from scipy.signal import welch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'meeting-01-10-2026',
                                'slides', 'code'))
from style import plt, BAND, BAD, BLUE, ORANGE   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'figures')
LOGS = os.path.join(HERE, '..', '..', 'server', 'initial-g')
FS = 4000.0
SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
BANDS = [(0, 20), (20, 106), (106, 140), (140, 230), (230, 297), (297, 2000.1)]


def band_rms(e, fs):
    N = e.shape[0]
    X = np.fft.rfft(e, axis=0); f = np.fft.rfftfreq(N, 1 / fs)
    pw = np.abs(X) ** 2 / N ** 2; pw[1:] *= 2
    if N % 2 == 0:
        pw[-1] /= 2
    return np.array([np.sqrt(pw[(f >= lo) & (f < hi)].sum(0)) for lo, hi in BANDS])


def val_curve(path):
    txt = open(path, encoding='utf-8', errors='replace').read().replace('\r', '\n')
    v0 = float(re.search(r'Initial Validation sim-RMS= ([0-9.e+-]+)', txt).group(1))
    pts = [(int(a), float(b)) for a, b in re.findall(r'\nIt (\d+), sqrt loss [0-9.e+-]+, Val sim-RMS ([0-9.e+-]+)', txt)]
    return np.array([(0, v0)] + pts)


def main(sp):
    os.makedirs(OUT, exist_ok=True)
    R = {k: np.load(os.path.join(sp, 'resid_%s.npz' % k)) for k in ('baseline', 'plain', 'ps2')}
    names = [str(n).split('_')[0] for n in R['ps2']['names']]
    ms = [i for i, n in enumerate(names) if not n.startswith('VA-T')]
    qt = [i for i, n in enumerate(names) if n.startswith('VA-T')]
    print('records', names, 'multisine', ms, 'quiet', qt)
    asd, stats = {}, {}
    for k, d in R.items():
        e = d['e']                                      # (rec, N, 3)
        f, P = welch(e[ms], fs=FS, window='hann', nperseg=4096, noverlap=2048, axis=1)
        asd[k] = np.sqrt(P.mean(0))                     # (F, 3)
        br = np.stack([band_rms(x, FS) for x in e])     # (rec, band, ch)
        whole = np.sqrt((e ** 2).mean(1))               # (rec, ch)
        stats[k] = dict(agg=d['agg'], whole_ms=whole[ms].mean(0), whole_qt=whole[qt].mean(0),
                        band_ms=br[ms].mean(0), band_qt=br[qt].mean(0))
        # consistency: Welch PSD integrated over 230-297 Hz vs Parseval band RMS (per-record mean of squares)
        m = (f >= 230) & (f < 297)
        welch_band = np.sqrt(P[:, m, 2].sum(1) * (f[1] - f[0])).mean()
        print('%-8s agg %s | whole Y ms %.3e qt %.3e | Y 230-297 Parseval %.3e Welch %.3e | Y 106-230 %.3e' % (
            k, np.array2string(np.atleast_1d(d['agg']), precision=3), stats[k]['whole_ms'][2], stats[k]['whole_qt'][2],
            stats[k]['band_ms'][4, 2], welch_band,
            np.sqrt(stats[k]['band_ms'][2, 2] ** 2 + stats[k]['band_ms'][3, 2] ** 2)))
        for b, (lo, hi) in enumerate(BANDS):
            print('    %4.0f-%-5.0f ms X1 %.2e X2 %.2e Y %.2e | quiet Y %.2e' % (
                lo, min(hi, 2000), *stats[k]['band_ms'][b], stats[k]['band_qt'][b, 2]))
    np.savez(os.path.join(OUT, 'live_resnet_spectrum_data.npz'), f=f, **{'asd_' + k: v for k, v in asd.items()})

    # same format as slide 3 (meeting-01-10 fig_absorber_issue.py): Y, 106 to 297 Hz, two shaded bands, % removed
    m = (f >= 107) & (f <= 296)
    lo_b = lambda k: np.sqrt(stats[k]['band_ms'][2, 2] ** 2 + stats[k]['band_ms'][3, 2] ** 2)
    removed = {k: (100 * (1 - lo_b(k) / lo_b('baseline')),
                   100 * (1 - stats[k]['band_ms'][4, 2] / stats['baseline']['band_ms'][4, 2])) for k in ('plain', 'ps2')}
    print('RMS removed vs baseline (106-230, 230-297):', removed)
    lo = min(asd[k][m, 2].min() for k in asd); hi = max(asd[k][m, 2].max() for k in asd)
    ylim = (lo / 8, hi * 8)
    sty = {'baseline': ('black', 'baseline error'), 'plain': (ORANGE, 'plain training'), 'ps2': (BLUE, 'with PS2')}
    for step, keys in (('a', ('baseline', 'plain')), ('b', ('baseline', 'plain', 'ps2'))):
        fig, ax = plt.subplots(figsize=(5.4, 4.9))
        ax.axvspan(106, 230, color=BAND, lw=0, zorder=0)
        ax.axvspan(230, 297, color=BAD, lw=0, zorder=0)
        for k in keys:
            ax.semilogy(f[m], asd[k][m, 2], color=sty[k][0], lw=2.2, label=sty[k][1])
        ax.set_xlim(106, 297)
        ax.set_ylim(*ylim)
        ax.minorticks_off()
        tk = [t for t in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5) if ylim[0] < t < ylim[1]]
        ax.set_yticks(tk)
        ax.set_yticklabels([('$10^{%d}$' % round(np.log10(t))) if abs(np.log10(t) - round(np.log10(t))) < 1e-6 else ('$3{ot}10^{%d}$' % np.floor(np.log10(t))) for t in tk])
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('Y error [m/$\sqrt{\mathrm{Hz}}$]')
        ax.legend(loc='upper left', frameon=False)
        y0 = ylim[0] * 1.15
        shown = [k for k in ('ps2', 'plain') if k in keys]
        for j, k in enumerate(shown):
            for x, val in ((168, removed[k][0]), (263, removed[k][1])):
                ax.text(x, y0 * 1.9 ** j, '%.0f %%' % val, ha='center', va='bottom', fontsize=13, fontweight='bold',
                        color=sty[k][0])
        ax.text(168, y0 * 1.9 ** len(shown), 'RMS removed', ha='center', va='bottom', fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, 'live_resnet_Y_error_asd_%s.png' % step), dpi=200)
        plt.close(fig)
    json_removed = {k: [round(v, 1) for v in removed[k]] for k in removed}

    cp = val_curve(os.path.join(LOGS, 'thesis-tcompare-resnet_88531_161.out'))
    cs = val_curve(os.path.join(LOGS, 'thesis-tcompare-resnet_88531_162.out'))
    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.semilogy(cp[:, 0], cp[:, 1], 'o-', color=ORANGE, label='plain training')
    ax.semilogy(cs[:, 0], cs[:, 1], 'o-', color=BLUE, label='with PS2')
    ax.axvline(2889, color=BLUE, ls=':', lw=1.4)
    ax.text(3100, cs[:, 1].min() * 0.93, 'NaN at update 2,889:\nbest weights kept', color=BLUE, fontsize=11, va='top')
    ax.set_xlabel('update')
    ax.set_ylabel('validation error, full records [m]')
    ax.set_ylim(cs[:, 1].min() * 0.6, cp[0, 1] * 1.3)
    ax.legend(loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'live_resnet_val_curve.png'), dpi=200)
    print('val plain', cp.tolist()); print('val ps2', cs.tolist())

    def sci(x):
        e = int(np.floor(np.log10(x))); return '%.2f·10%s' % (x / 10 ** e, str(e).translate(SUP))

    def lo_band(k):                                     # 106-230 Hz from the two sub-bands
        return np.sqrt(stats[k]['band_ms'][2, 2] ** 2 + stats[k]['band_ms'][3, 2] ** 2)

    def line(k):
        return 'Y RMS %s m overall, %s m in 230–297 Hz' % (sci(stats[k]['whole_ms'][2]), sci(stats[k]['band_ms'][4, 2]))

    pv = dict(cp[:, :2].tolist()); sv = dict(cs[:, :2].tolist())
    num = dict(
        val=dict(plain2600=sci(pv[2600]), ps22600=sci(sv[2600]), plain15600=sci(pv[15600]), ps2test=sci(2.375e-6)),
        removed=json_removed,
        side_base='Baseline: ' + line('baseline'),
        side_plain='Plain: ' + line('plain'),
        side_ps2='PS2: ' + line('ps2'),
        note_extra=('• side numbers: Y, mean over the four multisine records; 106-230 Hz: baseline %s, plain %s, PS2 %s m\n'
                    % (sci(lo_band('baseline')), sci(lo_band('plain')), sci(lo_band('ps2'))) +
                    '• full-record validation (all 6 records, all axes, mean per-record RMS): baseline %s, plain %s, '
                    'PS2 %s m; the PS2 value reproduces the logged 3.998e-06\n'
                    '• quiet records VA-T1, T2, Y whole RMS: baseline %s, plain %s, PS2 %s m\n'
                    '• ratios PS2 / plain on Y: whole %.2f, 106-230 Hz %.2f, 230-297 Hz %.2f' % (
                        sci(stats['baseline']['agg'].mean()), sci(stats['plain']['agg'].mean()),
                        sci(stats['ps2']['agg'].mean()), sci(stats['baseline']['whole_qt'][2]),
                        sci(stats['plain']['whole_qt'][2]), sci(stats['ps2']['whole_qt'][2]),
                        stats['ps2']['whole_ms'][2] / stats['plain']['whole_ms'][2], lo_band('ps2') / lo_band('plain'),
                        stats['ps2']['band_ms'][4, 2] / stats['plain']['band_ms'][4, 2])))
    json.dump(num, open(os.path.join(OUT, 'live_resnet_numbers.json'), 'w', encoding='utf-8'), ensure_ascii=False,
              indent=1)
    print(json.dumps(num, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
