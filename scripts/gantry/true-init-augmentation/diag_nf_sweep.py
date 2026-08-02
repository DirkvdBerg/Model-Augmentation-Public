"""How does the horizon change the balance between the IC ramp and the signal?

The per-window error from an exact 6-state seed is a RAMP: the missing absorber
momentum acts as a velocity deficit `(ma/mh)*vdelta_a(0)` which a `K = 0` axis
integrates, so the error grows like `t` while the thing the ANN is supposed to
learn, the absorber's ongoing force, contributes a bounded oscillation of about
`1e-05 m` on Y (coulomb-offset F2). The two therefore scale differently with the
horizon, and the ratio between them is a design quantity nobody has measured.

At `nf = 400` (0.100 s, the pipeline's `5*tau_msd`) the ramp reaches
`(ma/mh)*std(vdelta_a)*nf*ts = 2.2e-04 m`, about 20x the oscillation. At a short
horizon it would not dominate. That is a real tension with the reason `nf = 400`
was chosen (the absorber ring-down has to be visible inside the window), and this
sweep is what makes it quantitative rather than a hunch.

Two quantities per horizon, both against the free run at the SAME horizon so the
comparison is not confounded by the window length itself:

  DC ratio     per-window DC scatter, exact seed / free run
  RMS ratio    in-window RMS, exact seed / free run

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/diag_nf_sweep.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

from plant_cog import make_block, rollout_batch                     # noqa: E402
from data_exact import exact_truth                                  # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
REC = 'V1_standstill_Yp10'
NFS = (25, 50, 100, 200, 400, 800, 1600)
K0, N_WIN = 17, 200
STATES = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']


def main():
    tr = exact_truth(REC)
    rec, x6 = tr['rec'], tr['x6']
    ts, N = rec['ts'], len(x6)
    blk = make_block(Y_op=None, cog=True, ts=ts, up_sample=1, dtype=torch.float64)

    # one free run, reused for every horizon
    free = rollout_batch(blk, np.array([[0., 0., rec['Y_op'], 0., 0., 0.]]),
                         rec['u'][None, :N], n_out=6)[0]
    efree = free - x6

    print(f'Horizon sweep on the per-window target, {REC}, exact 6-state seed\n')
    print(f'  {"nf":>6}{"[ms]":>7}{"Y DC exact":>14}{"Y DC free":>13}{"DC ratio":>11}'
          f'{"Y RMS exact":>14}{"Y RMS free":>13}{"RMS ratio":>11}{"DC/RMS":>9}')
    res = {}
    for nf in NFS:
        span = N - nf - K0
        stride = max(1, span // N_WIN)
        starts = np.arange(K0, N - nf + 1, stride)[:N_WIN]
        u_win = np.stack([rec['u'][s:s + nf] for s in starts])
        ref = np.stack([x6[s:s + nf] for s in starts])
        sim = rollout_batch(blk, x6[starts], u_win, n_out=6)
        e = sim - ref
        ef = np.stack([efree[s:s + nf] for s in starts])
        dc_e, dc_f = e.mean(axis=1)[:, 2].std(), ef.mean(axis=1)[:, 2].std()
        rms_e = np.sqrt((e[:, :, 2] ** 2).mean())
        rms_f = np.sqrt((ef[:, :, 2] ** 2).mean())
        res[nf] = dict(ms=nf * ts * 1e3, dc_exact=float(dc_e), dc_free=float(dc_f),
                       rms_exact=float(rms_e), rms_free=float(rms_f),
                       dc_ratio=float(dc_e / dc_f), rms_ratio=float(rms_e / rms_f),
                       dc_over_rms=float(dc_e / rms_e),
                       all_states_dc=e.mean(axis=1).std(axis=0).tolist())
        print(f'  {nf:>6}{nf*ts*1e3:>7.1f}{dc_e:>14.4e}{dc_f:>13.4e}{dc_e/dc_f:>11.1f}'
              f'{rms_e:>14.4e}{rms_f:>13.4e}{rms_e/rms_f:>11.1f}{dc_e/rms_e:>9.3f}')

    print('\n  READING')
    print('   DC/RMS ~ sqrt(3)/2 = 0.866 is the signature of a pure ramp: the whole')
    print('   per-window error is the absorber initial condition and none of it is the')
    print('   signal. Where DC/RMS falls well below it, the oscillation the ANN is')
    print('   supposed to learn is a real share of the window.')
    print('   The RMS ratio is the honest "how much worse than a free run is this')
    print('   target" number at each horizon, and it is what a shorter nf would buy.')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_nf_sweep.json')
    with open(p, 'w') as f:
        json.dump(dict(record=REC, ts=float(ts), n_win=N_WIN, results=res), f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
