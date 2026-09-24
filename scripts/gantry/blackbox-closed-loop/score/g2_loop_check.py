"""G2 mechanism check: does the Tustin controller stabilise a KNOWN-GOOD linear plant at each rate?

If Cfb discretised at 800 Hz does not stabilise the true plant sampled at 800 Hz, the NaN of every
G2 checkpoint is a property of the controller at that rate, not of the models. Linear, frozen at
Y_op, no absorber motion nonlinearity; closed-loop poles of the discrete loop (plant ZOH at ts,
controller Tustin at ts, negative feedback, residual form = standard loop around the model).
Plants: the FP design plant of `gantry_dynamic/controller.py` (no absorber) and the old
`augmentation` truth of `msd_offset/plant.py` (ma_frac 0.10 absorber, 8 states).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

bbcl.bootstrap()
import numpy as np                                                         # noqa: E402
from gantry_dynamic import controller as C                                 # noqa: E402
sys.path.insert(0, os.path.join(bbcl.VENDOR, "msd_offset"))
import plant                                                               # noqa: E402
from bb_model import c2d_zoh                                               # noqa: E402


def fp_plant(Y):
    Minv = np.linalg.inv(C.M_op(Y))
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ C.K_STIFF, -Minv @ C.C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv @ C.P])          # stage force -> logical acceleration
    Cm = np.hstack([C.P.T, np.zeros((3, 3))])              # logical position -> stage position
    return A, B, Cm


def truth_old(Y):
    Minv = np.linalg.inv(plant.M8(Y, 0.0, freeze=True))
    A = np.block([[np.zeros((4, 4)), np.eye(4)], [-Minv @ plant._K4, -Minv @ plant._C4]])
    B = np.vstack([np.zeros((4, 3)), Minv @ plant._E43 @ plant.P_np])
    Cm = np.hstack([plant.P_np.T @ plant._E43.T, np.zeros((3, 4))])
    return A, B, Cm


def cl_poles(Ac, Bc, Cc, Y, ts):
    Ad, Bd = c2d_zoh(Ac, Bc, ts)
    Ak, Bk, Ck, Dk = C.controller_ss(Y, ts)
    n, m = Ad.shape[0], Ak.shape[0]
    # plant x+ = Ad x + Bd u, y = Cc x ; controller xc+ = Ak xc + Bk e, u = Ck xc + Dk e, e = -y
    Acl = np.block([[Ad - Bd @ Dk @ Cc, Bd @ Ck], [-Bk @ Cc, Ak]])
    return np.max(np.abs(np.linalg.eigvals(Acl)))


res = {}
for Y in (-0.22, 0.0):
    for fs in (800.0, 4000.0, 20000.0):
        r_fp = cl_poles(*fp_plant(Y), Y, 1 / fs)
        r_tr = cl_poles(*truth_old(Y), Y, 1 / fs)
        res['Y%+.2f_fs%d' % (Y, fs)] = dict(fp=float(r_fp), truth_old=float(r_tr))
        print('Y_op %+.2f  fs %6.0f Hz   max|z| closed loop: FP plant %.6f (%s)   old truth %.6f (%s)'
              % (Y, fs, r_fp, 'stable' if r_fp < 1 else 'UNSTABLE', r_tr,
                 'stable' if r_tr < 1 else 'UNSTABLE'))
os.makedirs(os.path.join(bbcl.OUT, 'g2'), exist_ok=True)
json.dump(res, open(os.path.join(bbcl.OUT, 'g2', 'loop_check.json'), 'w'), indent=1)
