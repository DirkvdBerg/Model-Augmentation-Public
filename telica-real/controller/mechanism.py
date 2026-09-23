"""G3 attempt 2: which mechanism explains the logged path M2 -> MF230 (diagnostic, TR-010).

(a) Row test: replay each BHL axis with each of the six zpk rows (num/den export, float64
    lfilter; fine for a diagnostic over <30k samples) and score how far |logged/replay| is from 1
    over the coherent band.
(b) Frame test: X1/X2 currents regressed on replays of BOTH X errors (logical-frame controller
    with a decoupling transform would need the cross terms); VAF with and without them.
(c) Empirical logged controller FRF per axis, H_e->i = S_ei / S_ee, against |K_zpk|.
Prints summaries only. Writes outputs/g3_mechanism/{mechanism.json, frf.png}.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.signal import lfilter, csd, welch, freqz, sosfreqz          # noqa: E402

import data_index                                                      # noqa: E402
from controller import telica_ctrl as tc                               # noqa: E402
from real_data_verification.telica_controller import _load_ba          # noqa: E402
from real_data_verification.telica_loader import load_telica_log_full  # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g3_mechanism')
AX = ('X1', 'X2', 'Y')
ROWS = ('LX1', 'LX2', 'LY', 'RX1', 'RX2', 'RY')
NPS = 4096


def main():
    ba, Ts = _load_ba()
    recs = [r for r in data_index.records(iters=('iter0',))]
    data = []
    for (s, op, it, p) in recs:
        d = load_telica_log_full(p)
        ok = (d['i_fb'].std(0) > 0) & (d['e'].std(0) > 0)
        data.append((op, d['e'], d['i_fb'], ok, d['motion_idx']))
    print(f'[mech] {len(data)} records loaded')

    # (c) empirical FRF e -> i and replay FRFs, accumulated over records (motion part)
    See = Sei = Sii = None
    for op, e, i, ok, k0 in data:
        f, a = welch(e[k0:], fs=1 / Ts, nperseg=NPS, axis=0)
        _, b = csd(e[k0:], i[k0:], fs=1 / Ts, nperseg=NPS, axis=0)
        _, c = welch(i[k0:], fs=1 / Ts, nperseg=NPS, axis=0)
        a = a * ok; b = b * ok; c = c * ok
        See = a if See is None else See + a
        Sei = b if Sei is None else Sei + b
        Sii = c if Sii is None else Sii + c
    Hlog = Sei / See
    coh = np.abs(Sei) ** 2 / (See * Sii)
    band = (f >= 5) & (f <= 5000)

    res = {'row_test': {}, 'frame_test': {}}
    # (a) row test: |Hlog / K_row| over coherent band
    print('\n(a) row test: median |log(|H_logged| / |K_row|)| over 5 Hz-5 kHz where coh > 0.9')
    print('      ' + ''.join(f'{r:>9s}' for r in ROWS))
    for j, ax in enumerate(AX):
        m = band & (coh[:, j] > 0.9)
        line = []
        for row in ROWS:
            bb, aa = ba[row]
            _, K = freqz(bb, aa, worN=f[m], fs=1 / Ts)
            dev = float(np.median(np.abs(np.log(np.abs(Hlog[m, j]) / np.abs(K)))))
            line.append(dev)
        res['row_test'][ax] = dict(zip(ROWS, line))
        print(f'  {ax:3s} ' + ''.join(f'{v:9.3f}' for v in line)
              + f'   best {ROWS[int(np.argmin(line))]}')

    # (b) frame test on X: i_j ~ K_j(e1), K_j(e2) (+ const) vs diagonal only
    print('\n(b) frame test (X only): VAF of i_Xj on [K_LXj(e_X1), K_LXj(e_X2), 1] vs diagonal')
    for j, ax in enumerate(('X1', 'X2')):
        v_d, v_f, coefs = [], [], []
        bb, aa = ba[('LX1', 'LX2')[j]]
        for op, e, i, ok, k0 in data:
            if not ok[j]:
                continue
            k1 = lfilter(bb, aa, e[:, 0]); k2 = lfilter(bb, aa, e[:, 1])
            y = i[:, j]; one = np.ones_like(y)
            Xd = np.column_stack([(k1, k2)[j], one])
            Xf = np.column_stack([k1, k2, one])
            cd, *_ = np.linalg.lstsq(Xd, y, rcond=None)
            cf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
            v_d.append(1 - (y - Xd @ cd).var() / y.var())
            v_f.append(1 - (y - Xf @ cf).var() / y.var())
            coefs.append(cf[:2].tolist())
        coefs = np.array(coefs)
        res['frame_test'][ax] = dict(vaf_diag=float(np.mean(v_d)), vaf_full=float(np.mean(v_f)),
                                     coef_mean=coefs.mean(0).tolist(), coef_std=coefs.std(0).tolist())
        print(f'  {ax}: VAF diag {np.mean(v_d):.4f}  full {np.mean(v_f):.4f}  '
              f'coef [own-rail, other-rail] mean {coefs.mean(0).round(3).tolist()} '
              f'std {coefs.std(0).round(3).tolist()}')

    # (c) summary of the empirical controller vs the zpk
    sos = tc.sos_bank()
    print('\n(c) |H_logged| / |K_zpk| and phase difference [deg] at selected frequencies')
    fsel = [2, 5, 10, 20, 50, 100, 150, 184, 200, 250, 300, 400, 600, 1000, 1500, 2000, 3000]
    res['frf_ratio'] = {}
    for j, ax in enumerate(AX):
        _, K = sosfreqz(sos[tc.AXES[j]], worN=f, fs=1 / Ts)
        ratio = Hlog[:, j] / K
        rows = []
        for fs_ in fsel:
            k = int(np.argmin(np.abs(f - fs_)))
            rows.append((fs_, float(np.abs(ratio[k])), float(np.degrees(np.angle(ratio[k]))),
                         float(coh[k, j])))
        res['frf_ratio'][ax] = rows
        print(f'  {ax}: ' + '  '.join(f'{a:g}Hz {b:.2f}/{c:+.0f}' for a, b, c, _ in rows))

    json.dump(res, open(os.path.join(OUT, 'mechanism.json'), 'w'), indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(AX):
        _, K = sosfreqz(sos[tc.AXES[j]], worN=f, fs=1 / Ts)
        axs[j, 0].loglog(f[1:], np.abs(Hlog[1:, j]), 'k', lw=0.8, label='logged e -> i (H1)')
        axs[j, 0].loglog(f[1:], np.abs(K[1:]), 'C3', lw=0.8, label=f'zpk {tc.AXES[j]}')
        axs[j, 0].set_ylabel(f'{ax} |K| [A/m]')
        axs[j, 1].semilogx(f[1:], np.degrees(np.angle(Hlog[1:, j])), 'k', lw=0.8)
        axs[j, 1].semilogx(f[1:], np.degrees(np.angle(K[1:])), 'C3', lw=0.8)
        axs[j, 1].set_ylabel('phase [deg]')
    axs[0, 0].legend(fontsize=8)
    axs[-1, 0].set_xlabel('f [Hz]'); axs[-1, 1].set_xlabel('f [Hz]')
    fig.suptitle('Empirical logged controller (iter0, all records) vs zpk file')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'frf.png'), dpi=110)
    np.savez(os.path.join(OUT, 'frf.npz'), f=f, Hlog=Hlog, coh=coh)


if __name__ == '__main__':
    main()
