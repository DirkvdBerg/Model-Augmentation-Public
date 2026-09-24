"""Absolute RMS [nm] of the baseline errors from the stored G4 replay metrics (no new simulation).

Per record and axis: rms(e_sim - e_meas) = NRMSE x rms(e_meas); medians over held-out records
(all iterations and iter0 only), plus the measured servo error itself.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402

VARS = [('rec_a2_5k', 'baseline + friction, 5 kHz (server default)'),
        ('rec_a2_nf_5k', 'same parameters, friction off, 5 kHz'),
        ('rec_a2', 'baseline + friction, 20 kHz'),
        ('nf', 'datasheet parameters, no friction, 20 kHz'),
        ('70821', '70821 parameters, no friction, 20 kHz')]
out = {}
for v, label in VARS:
    M = json.load(open(os.path.join(tr_env.OUTPUTS, f'g4_replay_{v}', 'metrics.json')))
    ho = [r for r in M['per_record'] if r['split'] != 'train']
    def med(rows, key):
        a = np.array([np.array(r[key]) * np.array(r['e_meas_rms_nm']) for r in rows])
        return np.nanmedian(a, axis=0)
    h0 = [r for r in ho if r['it'] == 'iter0']
    d, rf = med(ho, 'nrmse_direct'), med(ho, 'nrmse_resid')
    d0, rf0 = med(h0, 'nrmse_direct'), med(h0, 'nrmse_resid')
    meas = np.median([r['e_meas_rms_nm'] for r in ho], axis=0)
    meas0 = np.median([r['e_meas_rms_nm'] for r in h0], axis=0)
    out[v] = dict(direct=d.tolist(), resid=rf.tolist(), direct_iter0=d0.tolist(),
                  resid_iter0=rf0.tolist(), meas=meas.tolist(), meas_iter0=meas0.tolist())
    f = lambda a: ' / '.join(f'{x:6.0f}' for x in a)                   # noqa: E731
    print(f'{label}')
    print(f'   held-out all:  direct {f(d)}   residual form {f(rf)}   [nm, X1 / X2 / Y]')
    print(f'   held-out iter0: direct {f(d0)}   residual form {f(rf0)}')
print(f'measured servo error, held-out all: {" / ".join(f"{x:.0f}" for x in out["rec_a2_5k"]["meas"])} nm; '
      f'iter0: {" / ".join(f"{x:.0f}" for x in out["rec_a2_5k"]["meas_iter0"])} nm (5 kHz filtered); '
      f'20 kHz iter0: {" / ".join(f"{x:.0f}" for x in out["rec_a2"]["meas_iter0"])} nm')
json.dump(out, open(os.path.join(tr_env.out_dir('g4_rms_table'), 'rms_table.json'), 'w'), indent=1)
