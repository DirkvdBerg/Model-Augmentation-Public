"""Signal-level excitation audit of the simulated gantry datasets (read-only).

Per record, from the saved 20 kHz .mat files (no training, no generation):
  * injected multisine `f_sim` in LOGICAL channels [f_sym, f_anti (torque), f_Y]: RMS and the
    number of excited FFT lines per channel, and the band they occupy (audit D-207: amplitudes
    from the data, not the config);
  * where the total plant input `u_total` puts its power, per logical channel, in frequency bands
    (the baseline parameter families need different bands, see baseline-and-added-dynamics.md 8.1);
  * scheduling coverage: Y range, time share per 0.05 m bin, Y-rate statistics, and occupancy of
    a (Y, Ydot) grid (the LPV global-design question, E3);
  * absorber activation: RMS of delta_a and its share of power inside the multisine band (the
    informativity question for the learned component, E2).

Records: T1-T14, TP1-TP4 (train), V1-V4, VP1-VP2 (val), E1-E4, EP1 (test) of the frictionless
folder `augmentation_ma50_z03_b140-230_a6_telica`, whose T records are bit-identical (md5 checked
2026-09-24) to `augmentation_ma50_b140-230_a6_z03`, the set rho* = 0.205 was measured on.
Writes outputs/audit_signals.json and prints a table.
"""
import json
import os
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory',
                    'augmentation_ma50_z03_b140-230_a6_telica')
Lb = 0.725
P = np.array([[1, 1, 0], [Lb / 2, -Lb / 2, 0], [0, 0, 1]], dtype=np.float64)  # gtd_config.m
FAMILIES = {
    'standstill': ['T1_standstill_Ym30', 'T2_standstill_Ym15', 'T3_standstill_Y000',
                   'T4_standstill_Yp15', 'T5_standstill_Yp30'],
    'sweep': ['T6_ysweep_slow', 'T7_ysweep_fast', 'T8_ysweep_xmix'],
    'aprbs': ['T9_aprbs_30', 'T10_aprbs_60', 'T11_aprbs_100', 'T12_aprbs_yaw'],
    'lissajous': ['T13_lissajous', 'T14_lissajous_yaw'],
    'telica_profile': ['TP1_telica_y000', 'TP2_telica_y000_rev', 'TP3_telica_yp06',
                       'TP4_telica_ym06'],
    'val': ['V1_standstill_Yp10', 'V2_aprbs_Ylow', 'V3_ysweep_Yp10', 'V4_lissajous_Ym10',
            'VP1_telica_ylow', 'VP2_telica_yp12'],
    'test': ['E1_resonance_sweep', 'E2_multisine_Yp22', 'E3_aprbs_above', 'E4_multisine_off',
             'EP1_telica_test'],
}
BANDS = [(0, 1), (1, 10), (10, 100), (100, 140), (140, 230), (230, 1e9)]
# HEURISTIC: a line counts as excited if its power exceeds 1e-8 of the channel's largest line;
# random-phase flat multisines put every line at the same level, so any threshold far below 1
# and far above float32 round-off (about 1e-14 relative in power) gives the same count.
LINE_REL = 1e-8
Y_BIN = 0.05       # HEURISTIC: 0.05 m bins over the +-0.40 m stroke (lim.pos_Y, gtd_config.m)
YD_BIN = 0.10      # HEURISTIC: 0.10 m/s bins for the (Y, Ydot) occupancy grid


def spectrum(x, fs):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
    return f, np.abs(X) ** 2


def band_shares(f, pw):
    tot = pw[1:].sum()           # DC excluded: the band question is about dynamics
    return [float(pw[(f >= a) & (f < b)].sum() / tot) if tot > 0 else 0.0 for a, b in BANDS]


def one(name):
    d = loadmat(os.path.join(DATA, name + '.mat'), squeeze_me=True)
    fs = float(d['fs'])
    f_log = np.asarray(d['f_sim'], np.float64) @ P.T          # logical generalised forces
    u_log = np.asarray(d['u_total'], np.float64) @ P.T
    Y = np.asarray(d['Y_trajectory'], np.float64)
    da = np.asarray(d['delta_a'], np.float64)
    out = dict(name=name, fs=fs, N=int(len(Y)), amp_rms_saved=np.atleast_1d(d['amp_rms']).tolist())
    # multisine lines
    f, pw = spectrum(f_log, fs)
    lines, bands, rms = [], [], []
    for c in range(3):
        rms.append(float(np.sqrt(np.mean(f_log[:, c] ** 2))))
        mx = pw[1:, c].max()
        if mx <= 0:
            lines.append(0)
            bands.append([None, None])
            continue
        on = np.where(pw[:, c] > LINE_REL * mx)[0]
        on = on[on > 0]
        lines.append(int(len(on)))
        bands.append([float(f[on].min()), float(f[on].max())])
    out.update(ms_rms_logical=rms, ms_lines=lines, ms_band=bands,
               ms_df=float(fs / len(Y)))
    # where u_total puts its power
    fu, pu = spectrum(u_log - u_log.mean(0), fs)
    out['u_band_share'] = [band_shares(fu, pu[:, c]) for c in range(3)]
    out['u_rms_logical'] = [float(np.sqrt(np.mean((u_log[:, c] - u_log[:, c].mean()) ** 2)))
                            for c in range(3)]
    # scheduling coverage
    Yd = np.gradient(Y, 1.0 / fs)
    edges = np.arange(-0.40, 0.40 + 1e-9, Y_BIN)
    h, _ = np.histogram(Y, edges)
    out.update(Y_min=float(Y.min()), Y_max=float(Y.max()), Y_std=float(Y.std()),
               Y_bins_over_1pct=int((h / len(Y) > 0.01).sum()),
               Yd_rms=float(np.sqrt(np.mean(Yd ** 2))), Yd_absmax=float(np.abs(Yd).max()),
               Y_hist=(h / len(Y)).tolist())
    yd_edges = np.arange(-2.0, 2.0 + 1e-9, YD_BIN)
    H, _, _ = np.histogram2d(Y, Yd, [edges, yd_edges])
    out['Y_Yd_cells'] = H > 0
    # absorber activation
    fa, pa = spectrum(da - da.mean(), fs)
    tot = pa[1:].sum()
    out.update(da_rms=float(np.sqrt(np.mean((da - da.mean()) ** 2))),
               da_share_140_230=float(pa[(fa >= 140) & (fa <= 230)].sum() / tot) if tot > 0 else 0.0,
               da_mean=float(da.mean()))
    return out


def main():
    res, fam_cells = {}, {}
    for fam, names in FAMILIES.items():
        cells = None
        for n in names:
            r = one(n)
            c = r.pop('Y_Yd_cells')
            cells = c if cells is None else (cells | c)
            r['Y_Yd_cells_n'] = int(c.sum())
            res[n] = dict(family=fam, **r)
            print('%-22s ms_rms[sym,anti,Y]=%s lines=%s band=%s  u_share140-230=%s  '
                  'Y[%.3f,%.3f] bins>1%%=%d Ydrms=%.3f cells=%d  da_rms=%.3e da_in_band=%.3f'
                  % (n, ['%.1f' % v for v in r['ms_rms_logical']], r['ms_lines'],
                     [('%.1f-%.1f' % tuple(b)) if b[0] is not None else '-' for b in r['ms_band']],
                     ['%.3f' % s[4] for s in r['u_band_share']],
                     r['Y_min'], r['Y_max'], r['Y_bins_over_1pct'], r['Yd_rms'],
                     r['Y_Yd_cells_n'], r['da_rms'], r['da_share_140_230']), flush=True)
        fam_cells[fam] = int(cells.sum())
    train = [n for f in ('standstill', 'sweep', 'aprbs', 'lissajous', 'telica_profile')
             for n in FAMILIES[f]]
    # union of Y occupancy over the training set, time-weighted equally per record
    hist = np.mean([res[n]['Y_hist'] for n in train], axis=0)
    print('\nY occupancy over the 18 training records (share of time per 0.05 m bin, -0.40..0.40):')
    print('  ' + ' '.join('%.3f' % v for v in hist))
    print('(Y, Ydot) cells occupied per family:', fam_cells)
    with open(os.path.join(HERE, 'outputs', 'audit_signals.json'), 'w') as fh:
        json.dump(dict(records=res, family_cells=fam_cells, train_Y_hist=hist.tolist(),
                       bands=BANDS, y_bin=Y_BIN, yd_bin=YD_BIN), fh, indent=1)
    print('[out] outputs/audit_signals.json')


if __name__ == '__main__':
    sys.exit(main())
