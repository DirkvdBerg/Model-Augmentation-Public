"""G4 step 0b: holding forces before and after the single move (friction memory test).

Hypothesis (to test, not assume): a rail at rest is held by stiction, so the controller's
integrator keeps whatever force it had when the rail stopped: approached moving negative before
the log -> F_start ~ F_ext - cc; stopped after the + move -> F_end ~ F_ext + cc. If so,
cc ~ (F_end - F_start)/2 and the static external force F_ext ~ (F_end + F_start)/2, separable
although every record moves one way only. Uses the TOTAL applied force (MF30 x Kt s_F).
Prints summaries; writes outputs/g4_holding/holding.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402
from scipy.signal import butter, sosfiltfilt                           # noqa: E402

from params import telica_params as tp                                 # noqa: E402
from baseline.records import iter_records                              # noqa: E402
tr_env.check_no_leak()

OUT = tr_env.out_dir('g4_holding')
LP = butter(4, 200.0, fs=tp.FS, output='sos')
AX = ('X1', 'X2', 'Y')


def main():
    rows = []
    for s, op, it, d in iter_records():
        q, u, k0 = d['q1'], d['u'], d['motion_idx']
        v = np.gradient(sosfiltfilt(LP, q, axis=0), 1 / tp.FS, axis=0)
        moving = np.abs(v) >= tp.V_BRK
        last_move = int(np.where(moving.any(1))[0].max())
        tail = slice(last_move + 2000, None)                  # >= 100 ms after the last sliding
        if len(u) - (last_move + 2000) < 400:
            continue
        rest_ok = bool(np.abs(v[tail]).max() < tp.V_BRK)
        F0 = u[:k0 - 200].mean(0); F1 = u[tail].mean(0)
        rows.append(dict(split=s, op=op, it=it, F_start=F0.tolist(), F_end=F1.tolist(),
                         tail_len=int(len(u) - last_move - 2000), rest_ok=rest_ok,
                         ff_end=(d['u'] - d['u_fb'])[tail].mean(0).tolist()))
    json.dump(rows, open(os.path.join(OUT, 'holding.json'), 'w'), indent=1)
    R = [r for r in rows if r['rest_ok']]
    print(f'[hold] {len(rows)} records with a >= 20 ms dwell after the move, {len(R)} at rest there')
    sF = np.asarray(tp.S_FORCE['A'])
    for j, ax in enumerate(AX):
        F0 = np.array([r['F_start'][j] for r in R]); F1 = np.array([r['F_end'][j] for r in R])
        cc = (F1 - F0) / 2; fx = (F1 + F0) / 2
        ff = np.array([r['ff_end'][j] for r in R])
        print(f'[hold] {ax}: F_start median {np.median(F0):7.1f} N (IQR {np.percentile(F0, 25):.1f}'
              f'..{np.percentile(F0, 75):.1f}), F_end median {np.median(F1):7.1f} N '
              f'(IQR {np.percentile(F1, 25):.1f}..{np.percentile(F1, 75):.1f}); sign flips in '
              f'{int(np.sum(np.sign(F0) != np.sign(F1)))}/{len(R)}')
        print(f'[hold] {ax}: cc_est = (F_end-F_start)/2 median {np.median(cc):6.1f} N '
              f'(IQR {np.percentile(cc, 25):.1f}..{np.percentile(cc, 75):.1f}) [reading B: '
              f'{np.median(cc) / sF[j]:.1f}]; F_ext_est median {np.median(fx):6.1f} N '
              f'(IQR {np.percentile(fx, 25):.1f}..{np.percentile(fx, 75):.1f}) [B: '
              f'{np.median(fx) / sF[j]:.1f}]; datasheet static max {tp.CC[j]:.0f} N; '
              f'FF share of F_end median {np.median(ff):.1f} N')
        by = {}
        for r, c in zip(R, cc):
            by.setdefault(r['split'], []).append(c)
        print(f'[hold] {ax}: cc_est by split ' + ', '.join(f'{k} {np.median(v):.1f} (n={len(v)})'
                                                          for k, v in by.items()))


if __name__ == '__main__':
    main()
