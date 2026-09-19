"""The only script here that simulates. Writes `settling_data.npz`; the figures redraw from it.

    python collect.py

Four arms (baseline, no projection, OBC tangent, OBC affine) on ONE record,
`S1_settle_Ylow_noexc`, in the closed loop. Each arm is also re-simulated on `V2_aprbs_Ylow`,
whose per-record NRMS the training log already printed, so "the right weights went through the
right loop" is CHECKED rather than claimed. For run 84033 the log line to compare against is

    V2_aprbs_Ylow.mat: [ 26.739594  36.532574 703.3376  ]

and for 84032

    (its own log, same banner: "Validation-set NRMS per record (augmented, avg from K0)")

A mismatch here means the config replay or the borrowed controller row is wrong, and the figure
must not be drawn. It is printed, not asserted, because the baseline arm has no logged
counterpart and because a tolerance would be one more number picked by hand.

The reference `r_sim` is carried through as well, decimated the same way `load_traj` decimates
`y`, because the settling windows are located from the SETPOINT. They cannot be located from `y`:
`y` never stops moving, the absorber rings through the dwell, and that ringing is the figure.
"""
__project_origin__ = "added"

import sys
import time

import numpy as np
import scipy.io as sio

from fig_common import (ARM_ORDER, ARMS, FS, RECORD, RECORD_MAT, SETTLE_MODE, TRAIN_MODE,
                    VERIFY_RECORD, cache_path)
from build import build_arm, closed_loop_sim, controller_row, load_record, nrms


def _reference(cfg):
    """`r_sim` from the record's .mat, decimated by point sampling, exactly as `y` is."""
    import os
    from gantry_dynamic.config import REPO_ROOT
    p = os.path.join(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', SETTLE_MODE, RECORD_MAT)
    d = sio.loadmat(p, squeeze_me=True)
    if np.abs(d['f_sim']).max() != 0.0:
        raise SystemExit('%s carries a nonzero injected force (max |f_sim| = %.3e). This figure '
                         'is about the multisine-free record; refusing to draw it on a record '
                         'that has one.' % (RECORD_MAT, np.abs(d['f_sim']).max()))
    return d['r_sim'][::cfg.d].astype(np.float64), d['delta_a'][::cfg.d].astype(np.float64)


def main():
    t_start = time.time()
    out = {}
    r_dec = da_dec = None
    k0_seen = None

    for arm in ARM_ORDER:
        print('\n' + '=' * 92)
        print('--- arm %r: run %s, weights=%s ---' % (arm, ARMS[arm]['run'], ARMS[arm]['weights']))
        print('=' * 92)
        fit_sys, data, norm, cfg = build_arm(arm)
        row = controller_row(fit_sys)
        print('[ctrl] borrowing the Y_op = -0.22 controller row from V2: row %d' % row)

        # 1. the verification pass, on a record whose number the training log already holds
        sd_v2 = load_record(cfg, VERIFY_RECORD, TRAIN_MODE)
        yh_v2, yt_v2, k0 = closed_loop_sim(fit_sys, sd_v2, row)
        print('[check] %s NRMS = %s' % (VERIFY_RECORD, np.array2string(
            nrms(yh_v2, yt_v2, norm), formatter={'float_kind': lambda v: '%10.6f' % v})))

        # 2. the record the figures are about
        sd = load_record(cfg, RECORD_MAT, SETTLE_MODE)
        yh, yt, k0b = closed_loop_sim(fit_sys, sd, row)
        if k0b != k0:
            raise RuntimeError('k0 differs between records within one arm (%d vs %d)' % (k0, k0b))
        print('[sim]   %s NRMS = %s' % (RECORD, np.array2string(
            nrms(yh, yt, norm), formatter={'float_kind': lambda v: '%10.6f' % v})))
        print('[sim]   %s rms error [m] = %s' % (RECORD, np.array2string(
            np.sqrt(((yh - yt) ** 2).mean(axis=0)),
            formatter={'float_kind': lambda v: '%.4e' % v})))

        if k0_seen is None:
            k0_seen = k0
            r_full, da_full = _reference(cfg)
            r_dec, da_dec = r_full[k0:][:len(yt)], da_full[k0:][:len(yt)]
            out['y_true'] = yt
            out['r'] = r_dec
            out['delta_a'] = da_dec
            out['k0'] = k0
            out['fs'] = FS
        elif k0 != k0_seen:
            raise RuntimeError('arms disagree on k0 (%d vs %d); they are not the same loop'
                               % (k0, k0_seen))
        else:
            if not np.array_equal(out['y_true'], yt):
                raise RuntimeError('arms disagree on the data itself; something is loading a '
                                   'different record')

        out['yhat_' + arm] = yh
        out['nrms_' + arm] = nrms(yh, yt, norm)
        out['nrms_v2_' + arm] = nrms(yh_v2, yt_v2, norm)
        del fit_sys, data, norm

    out['arms'] = np.array(ARM_ORDER)
    out['record'] = np.array(RECORD)
    np.savez_compressed(cache_path(), **out)
    print('\nwrote %s  (%.1f s)' % (cache_path(), time.time() - t_start))
    print('now draw with:  python fig_settling.py')


if __name__ == '__main__':
    sys.exit(main())
