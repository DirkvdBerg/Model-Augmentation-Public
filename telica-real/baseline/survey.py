"""G4 step 0: what the Telica records contain (summaries only; no gate metric).

Per record: length, motion start, per-rail peak velocity/acceleration, share of sliding samples
by direction (can Coulomb be separated from a constant force?), standstill servo-error rms and
standstill mean current (holding force). Also checks the fast pandas parser against the
reference one on the first file.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.signal import butter, sosfiltfilt                           # noqa: E402

import data_index                                                      # noqa: E402
from params import telica_params as tp                                 # noqa: E402
from real_data_verification.telica_loader import load_telica_log_full  # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g4_survey')
FS = tp.FS
LP = butter(4, 200.0, fs=FS, output='sos')        # HEURISTIC: 200 Hz, see TR-014


def main():
    recs = data_index.records(optional=True, iters=tuple(data_index.ITERS_COMMON))
    print(f'[survey] {len(recs)} records')
    # parser check
    t0 = time.time(); a = load_telica_log_full(recs[0][3], engine='python'); t1 = time.time()
    b = load_telica_log_full(recs[0][3], engine='c'); t2 = time.time()
    same = all(np.array_equal(a[k], b[k]) for k in ('r', 'q1', 'e', 'i_fb', 'i_tot'))
    print(f'[survey] parser check: identical={same}, python {t1 - t0:.2f} s, c {t2 - t1:.2f} s')
    engine = 'c' if same else 'python'
    rows = []
    a2n = np.asarray(tp.a_to_n())
    from baseline.records import iter_records
    for (s, op, it, d) in iter_records():
        q, e, i = d['q1'], d['e'], d['i_tot']
        k0 = d['motion_idx']
        qf = sosfiltfilt(LP, q, axis=0)
        v = np.gradient(qf, 1 / FS, axis=0)
        acc = np.gradient(v, 1 / FS, axis=0)
        mo = slice(k0, None)
        slide = np.abs(v[mo]) >= tp.V_BRK
        pos = (v[mo] >= tp.V_BRK).mean(0); neg = (v[mo] <= -tp.V_BRK).mean(0)
        still = slice(0, max(k0 - 200, 1))
        row = dict(split=s, op=op, it=it, T=len(q), k0=int(k0), motion_s=(len(q) - k0) / FS,
                   vmax=np.abs(v[mo]).max(0).tolist(), amax=np.abs(acc[mo]).max(0).tolist(),
                   slide_frac=slide.mean(0).tolist(), pos_frac=pos.tolist(), neg_frac=neg.tolist(),
                   e_still_rms=e[still].std(0).tolist(), e_motion_rms=e[mo].std(0).tolist(),
                   F_still_mean=(i[still].mean(0) * a2n).tolist(),
                   F_motion_rms=(i[mo].std(0) * a2n).tolist(),
                   q_range=(q.max(0) - q.min(0)).tolist())
        rows.append(row)
    json.dump(dict(engine=engine, rows=rows), open(os.path.join(OUT, 'survey.json'), 'w'), indent=1)
    R = rows
    f = lambda k, j: np.array([r[k][j] for r in R])                       # noqa: E731
    print(f'[survey] T {min(r["T"] for r in R)}..{max(r["T"] for r in R)}, motion '
          f'{min(r["motion_s"] for r in R):.3f}..{max(r["motion_s"] for r in R):.3f} s, '
          f'k0 {min(r["k0"] for r in R)}..{max(r["k0"] for r in R)}')
    for j, ax in enumerate(('X1', 'X2', 'Y')):
        print(f'[survey] {ax}: vmax {f("vmax", j).min():.3f}..{f("vmax", j).max():.3f} m/s, '
              f'amax {f("amax", j).min():.1f}..{f("amax", j).max():.1f} m/s2, sliding '
              f'{f("slide_frac", j).mean():.3f} of motion (v>0 {f("pos_frac", j).mean():.3f}, '
              f'v<0 {f("neg_frac", j).mean():.3f}; records with both directions '
              f'{int(((f("pos_frac", j) > 0.02) & (f("neg_frac", j) > 0.02)).sum())}/{len(R)}), '
              f'range {f("q_range", j).min() * 1e3:.1f}..{f("q_range", j).max() * 1e3:.1f} mm')
        print(f'[survey] {ax}: standstill e rms {np.median(f("e_still_rms", j)) * 1e9:.1f} nm '
              f'(median), motion e rms {np.median(f("e_motion_rms", j)) * 1e9:.0f} nm, holding '
              f'force {f("F_still_mean", j).min():.1f}..{f("F_still_mean", j).max():.1f} N '
              f'(median {np.median(f("F_still_mean", j)):.1f}), motion force rms '
              f'{np.median(f("F_motion_rms", j)):.0f} N')
    by_it = {}
    for r in R:
        by_it.setdefault(r['it'], []).append(np.median(np.array(r['e_motion_rms'])))
    print('[survey] motion servo-error rms (median over axes) by iteration [nm]: '
          + ', '.join(f'{k} {np.median(v) * 1e9:.0f}' for k, v in sorted(by_it.items())))


if __name__ == '__main__':
    main()
