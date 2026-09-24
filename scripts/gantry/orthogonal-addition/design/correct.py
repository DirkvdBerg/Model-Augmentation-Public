"""Solve a stateless COMPENSATOR from the legitimate library so the ten conditions hold exactly.

Two targets (OA-011, handoff sect. 5):
  OA_TARGET=model     the addition's own one-step field from its definition (addition_io) on the
                      tuples of OA_MODE: Phi^T (Delta_add + Delta_comp) = 0. On the null control
                      this is the first-order SCREEN of an addition.
  OA_TARGET=measured  the dataset OA_MODE was generated WITH the current addition; its MEASURED
                      Delta* (stencil variant, as measure_delta.py) is used, and the floor's own
                      bias is kept: Phi^T (Delta* + Delta_comp) = Phi^T Phi b_null, b_null from
                      OA_NULL_JSON (the null control's J^+ Delta*). This re-solves on the
                      regenerated data, so the absorber's closed-loop effect is in the coefficients.
Among all compensators satisfying the ten conditions the one with the least stage-actuator force
energy on the tuples is taken (OA-009 metric). Output: OA_ADDITION + compensator -> OA_OUT (.mat).
Env: OA_MODE, OA_ADDITION, OA_OUT, OA_TARGET, OA_NULL_JSON, OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES               # noqa: E402
import columns as colib                                # noqa: E402
import addition_io as aio                              # noqa: E402


def null_bias(path):
    js = json.load(open(path))
    p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
    return np.asarray(p['dtheta'], float), np.asarray(p['pct'], float)


def main():
    pb = Problem()
    target = os.environ.get('OA_TARGET', 'model')
    add_path, out_path = os.environ['OA_ADDITION'], os.environ['OA_OUT']
    oa = aio.load_mat(add_path)
    fa = aio.force_field(pb, oa)
    Dmodel = pb.delta_of(fa)
    if target == 'model':
        D0 = Dmodel
        t = np.zeros(10)
    elif target == 'measured':
        D0 = pb.measured_delta()
        b_null, b_null_pct = null_bias(os.environ['OA_NULL_JSON'])
        bt = torch.from_numpy(b_null)
        t = pb.basis.jt(pb.basis.apply(bt)).numpy()
        Dm_only = D0 - Dmodel
        print('[corr] measured ||Delta*|| %.4e; model field of the addition %.4e; '
              'measured minus model %.4e (floor + model error)'
              % (np.linalg.norm(D0), np.linalg.norm(Dmodel), np.linalg.norm(Dm_only)))
    else:
        raise ValueError(target)
    j0 = pb.stats(D0)[0]
    pct0, own0 = pb.bias_pct(D0)
    print('[corr] %s target on %s (stride %d): rho* %.4e  ||Delta|| %.4e  bias max|.| %.3e %%  '
          'm_diff own %+.3e %%' % (target, pb.mode, pb.stride, pb.rho(D0), np.linalg.norm(D0),
                                   np.abs(pct0).max(), own0))

    lib = colib.stateless_library(max_ypow=2)
    if os.environ.get('OA_LIB', 'KM') == 'K':
        # OA-012: stiffness slots only (17 columns): they act where the controller is stiff
        lib = [s for s in lib if s[0] == 'K']
    elif os.environ.get('OA_LIB', 'KM') == 'M':
        # OA-018: inertial slots only (11 columns up to Y^2): forces follow qdd, i.e. the in-band
        # multisine the loop does not cancel, so the recorded u does not carry them
        lib = [s for s in lib if s[0] == 'M']
    print('[corr] compensator library: %d slots (%s)' % (len(lib), os.environ.get('OA_LIB', 'KM')))
    A = np.zeros((10, len(lib))); nrm = np.zeros(len(lib)); Fs = []; Dc = []
    obj = os.environ.get('OA_OBJ', 'force')
    for c, s in enumerate(lib):
        f = pb.stateless_force(s[0], s[1], s[2], s[3])
        D = pb.delta_of(f)
        A[:, c], _u, nrm[c] = pb.stats(D)
        if obj == 'force':
            Fs.append((f @ pb.Pinv.T).reshape(-1).astype(np.float32))
        else:
            # OA-013: least ||Delta_comp||, so the absorber keeps the largest possible share
            Fs.append(D.astype(np.float32))
    Fs = np.stack(Fs, axis=1)
    Fg = (Fs.T @ Fs).astype(np.float64) / np.outer(nrm, nrm)
    del Fs
    print('[corr] compensator objective: least %s' % ('stage force' if obj == 'force' else '||Delta_comp||'))
    An = A / nrm[None, :]
    n = len(lib)
    KKT = np.block([[2 * Fg, An.T], [An, np.zeros((10, 10))]])
    rhs = np.concatenate([np.zeros(n), t - j0])
    z = np.linalg.solve(KKT, rhs)[:n]
    c = z / nrm
    comp = aio.empty()
    for ci, s in zip(c, lib):
        aio.add_slot(comp, s, ci)
    fc = aio.force_field(pb, comp)
    Dcomp = pb.delta_of(fc)
    D1 = D0 + Dcomp
    j1 = pb.stats(D1)[0]
    pct1, own1 = pb.bias_pct(D1)
    resid = np.linalg.norm(j1 - t) / max(np.linalg.norm(j0 - t), 1e-300)
    print('[corr] after compensator: ||Phi^T D - target|| reduced by %.2e; rho* %.4e  '
          '||Delta|| %.4e' % (resid, pb.rho(D1), np.linalg.norm(D1)))
    print('[corr] predicted bias (combo %%): %s' % '  '.join('%s %+.3e' % (nm, p) for nm, p in
                                                             zip(COMBO_NAMES, pct1)))
    print('[corr] m_diff own scale %+.3e %%' % own1)
    fcs = fc @ pb.Pinv.T
    fas = fa @ pb.Pinv.T
    print('[corr] compensator stage force rms %s N peak %s N | addition before: rms %s N peak %s N'
          % (np.array2string(np.sqrt((fcs ** 2).mean(0)), precision=2),
             np.array2string(np.abs(fcs).max(0), precision=1),
             np.array2string(np.sqrt((fas ** 2).mean(0)), precision=2),
             np.array2string(np.abs(fas).max(0), precision=1)))
    print('[corr] ||Delta_comp|| %.4e = %.1f %% of ||Delta_addition + Delta_comp|| (model)'
          % (np.linalg.norm(Dcomp), 100 * np.linalg.norm(Dcomp) / np.linalg.norm(Dmodel + Dcomp)))
    print('[corr] compensator coefficients:')
    for ci, s in zip(c, lib):
        print('    %-8s %+.6e' % (colib.slot_name(s), ci))
    oa2 = {k: v.copy() for k, v in oa.items()}
    for ci, s in zip(c, lib):
        aio.add_slot(oa2, s, ci)
    aio.save_mat(out_path, oa2, '%s + compensator (%s target on %s)'
                 % (os.path.basename(add_path), target, pb.mode))
    with open(out_path.replace('.mat', '.json'), 'w') as fh:
        json.dump(dict(mode=pb.mode, stride=pb.stride, target=target,
                       before=dict(rho=pb.rho(D0), delta=float(np.linalg.norm(D0)),
                                   bias_pct=pct0.tolist(), m_diff_own=own0),
                       after=dict(rho=pb.rho(D1), delta=float(np.linalg.norm(D1)),
                                  bias_pct=pct1.tolist(), m_diff_own=own1),
                       comp_force_rms=np.sqrt((fcs ** 2).mean(0)).tolist(),
                       comp_force_peak=np.abs(fcs).max(0).tolist(),
                       comp_delta=float(np.linalg.norm(Dcomp)),
                       compensator={colib.slot_name(s): float(ci) for ci, s in zip(c, lib)}),
                  fh, indent=2)
    print('[out] %s' % out_path)


if __name__ == '__main__':
    main()
