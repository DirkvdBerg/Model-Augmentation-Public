"""G2 screen, route A (parity): an EVEN addition against the odd, state-linear regressor.

Gyorok Sect. 5.1 template: with a sign-symmetric dataset (every tuple (x, u) paired with
(-x, -u) about the operating point) an even addition is orthogonal EXACTLY (Eq. 16; d_symbolic
D10). On the gantry the scheduling variable Y must stay fixed under the mirror (M(Y)), so only the
STANDSTILL records (T1 to T5, Y parked, X and Theta at 0, a zero-mean multisine) can be mirrored:
the partner record is the same reference with the multisine negated. The screen therefore asks
the deciding question first: on those records, how large is Delta for a physically sized even
addition, against the OA-004 minimum?

Candidates, all even in (deviation state, input):
  kappa  position-dependent force constant, f = kappa X u_log, kappa = 1/0.3 m^-1 (the force
         constant changing by 100 % over the 0.3 m half stroke: a generous UPPER bound)
  cor    Coriolis/centripetal forces the frozen-M baseline drops, leading terms from the
         Christoffel symbols of M(Y): f = mh [-2 Thetad Yd, 2 Y Thetad Yd, Y Thetad^2]
         (order-of-magnitude form; the exact expression is gantrySystemCoriolisCentripetal.m)
  const  a constant force F0 = 10 N on X and Y (cable-carrier preload): even, exactly orthogonal
         on mirrored pairs, but it is not dynamics
Env: OA_MODE, OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, OA                         # noqa: E402

DELTA_MIN_STRIDE1 = 8.49          # OA-004


def main():
    pb = Problem()
    from model_augmentation.systems import gantry_ss as gss
    mh = float(gss.mh)
    P = np.linalg.inv(pb.Pinv)
    u_log = pb.ref.u_stage.numpy() @ P.T                 # stage force -> logical force
    q, qd = pb.q, pb.qd
    rec_ix = pb.ref.record_ix.numpy()
    stand = rec_ix < 5                                   # T1..T5
    cands = {}
    kappa = 1.0 / 0.3
    f = np.zeros_like(q); f[:] = kappa * q[:, [0]] * u_log
    cands['kappa'] = f
    f = np.zeros_like(q)
    f[:, 0] = -2.0 * mh * qd[:, 1] * qd[:, 2]
    f[:, 1] = 2.0 * mh * q[:, 2] * qd[:, 1] * qd[:, 2]
    f[:, 2] = mh * q[:, 2] * qd[:, 1] ** 2
    cands['cor'] = f
    f = np.zeros_like(q); f[:, 0] = 10.0; f[:, 2] = 10.0
    cands['const'] = f
    n6 = len(pb.ref.record_ix) * 6
    mask = np.repeat(stand, 6)
    dmin_full = DELTA_MIN_STRIDE1 / np.sqrt(pb.stride)
    dmin_stand = dmin_full * np.sqrt(stand.mean())
    out = {}
    print('standstill tuples %d of %d; minimum ||Delta|| on them (pro rata) %.3e'
          % (stand.sum(), len(stand), dmin_stand))
    print('X rms on standstill %.3e m, Thetad rms %.3e rad/s, Yd rms %.3e m/s, |u_log| rms %s'
          % (np.sqrt((q[stand, 0] ** 2).mean()), np.sqrt((qd[stand, 1] ** 2).mean()),
             np.sqrt((qd[stand, 2] ** 2).mean()),
             np.array2string(np.sqrt((u_log[stand] ** 2).mean(0)), precision=1)))
    for k, f in cands.items():
        D = pb.delta_of(f)
        Ds = np.where(mask, D, 0.0)
        ns = float(np.linalg.norm(Ds))
        out[k] = dict(norm_standstill=ns, ratio_to_min=ns / dmin_stand,
                      rho_full_unmirrored=pb.rho(D), norm_full=float(np.linalg.norm(D)))
        print('  %-6s ||Delta|| on standstill %.3e = %.2e x the minimum;  full set %.3e, '
              'rho* unmirrored %.3f' % (k, ns, ns / dmin_stand, out[k]['norm_full'],
                                        out[k]['rho_full_unmirrored']))
    with open(os.path.join(OA, 'outputs', 'screen_a_%s_s%d.json' % (pb.mode, pb.stride)), 'w') as fh:
        json.dump(out, fh, indent=2)


if __name__ == '__main__':
    main()
