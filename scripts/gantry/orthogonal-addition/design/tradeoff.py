"""OA-020: predicted G4 shift against addition size, u-aware (validated in R-020 against R-019).

For an addition file and a list of sizes s: scale it so that ||G phi|| = s, evaluate the Jacobian
at the shifted input (x, u - phi) on the null-control tuples, and predict the shift of J'^+ (Delta*_null
+ G phi) against b_null. Prints one line per size.
Env: OA_MODE (null control), OA_ADDITION, OA_NULL_JSON, OA_SIZES (comma list), OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import gc
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES               # noqa: E402
import addition_io as aio                              # noqa: E402
from design_quadratic import jac, shifts               # noqa: E402


def main():
    pb = Problem()
    pb.basis = None; gc.collect()
    js = json.load(open(os.environ['OA_NULL_JSON']))
    p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
    b = np.asarray(p['dtheta'], float)
    Dnull = pb.measured_delta()
    oa = aio.load_mat(os.environ['OA_ADDITION'])
    f1 = aio.force_field(pb, oa)
    D1 = pb.delta_of(f1)
    s1 = float(np.linalg.norm(D1))
    sizes = [float(s) for s in os.environ.get('OA_SIZES', '1,2,4,8.49,12.735').split(',')]
    out = []
    print('[trade] %s: native size %.4f' % (os.path.basename(os.environ['OA_ADDITION']), s1))
    for s in sizes:
        a = s / s1
        J = jac(pb, a * f1)
        mx, own, sh = shifts(pb, J, Dnull + a * D1, b)
        del J; gc.collect()
        fs = a * (f1 @ pb.Pinv.T)
        print('[trade] size %7.3f (x%.4f): predicted max shift nine %.4f %%, m_diff own %+.4f %%, '
              'force rms %s N' % (s, a, mx, own, np.array2string(np.sqrt((fs ** 2).mean(0)), precision=2)),
              flush=True)
        out.append(dict(size=s, scale=a, max_shift=mx, m_diff_own=own, shifts=sh.tolist()))
    tag = os.path.splitext(os.path.basename(os.environ['OA_ADDITION']))[0]
    json.dump(out, open(os.path.join(HERE, '..', 'outputs', 'tradeoff_%s.json' % tag), 'w'), indent=2)


if __name__ == '__main__':
    main()
