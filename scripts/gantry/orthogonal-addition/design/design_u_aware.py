"""Second-order-aware re-solve of a stateless addition (OA-014): Phi evaluated at the SHIFTED input.

Mechanism (R-017): the loop is stiff below its 100 Hz bandwidth, so a low-frequency addition force
phi(x) is cancelled by the controller: the regenerated record keeps (almost) the same x and
records u_new = u - phi (stage: u - P^-1 phi). Because the production step is affine in u,
    Delta*_new = x_next - f_base(x, u - phi) = Delta*_null + G phi        (exact, what the design used)
but the parameter sensitivity is evaluated at (x, u - phi), which the null-control design did not.
The ten conditions of the G4 target, J'^+ (Delta*_null + G phi) = b_null, i.e.
    J'^T (Delta*_null + G phi(kappa)) = J'^T J' b_null,     J' = Phi(x, u - phi(kappa)),
are quadratic in the addition's coefficients kappa. Solved by iteration: J' at the current kappa,
then the least-force correction delta (KKT, the OA-009 metric) that zeroes the linearised
residual, until the residual is at round-off. The addition's size ||G phi|| is monitored.
Env: OA_MODE (null control), OA_ADDITION (start .mat, stateless), OA_OUT, OA_NULL_JSON,
     OA_ITERS (default 6), OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES, ROWS          # noqa: E402
import columns as colib                                # noqa: E402
import addition_io as aio                              # noqa: E402
from vendor.obc import OBCBasis, build_stacked_sensitivity   # noqa: E402


def basis_at(pb, f_log):
    """Basis of Phi at the tuples with the stage input shifted by -P^-1 f_log."""
    Z = pb.ref.Z_step.clone().to(torch.float64)
    Z[:, 6:9, 0] -= torch.from_numpy((f_log @ pb.Pinv.T) / pb.std_u)
    J = build_stacked_sensitivity(pb.step, pb.vbar.to(pb.device), Z, ROWS,
                                  chunk=pb.cfg.obc_ref_chunk, device=pb.device)
    b = OBCBasis.from_matrix(J.cpu(), pb.vbar.cpu())
    del J, Z
    if pb.device.type == 'cuda':
        torch.cuda.empty_cache()
    return b


def main():
    pb = Problem()
    oa0 = aio.load_mat(os.environ['OA_ADDITION'])
    assert np.all(oa0['oa_m'] == 0), 'stateless additions only'
    js = json.load(open(os.environ['OA_NULL_JSON']))
    p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
    b_null = torch.tensor(p['dtheta'], dtype=torch.float64)
    Dnull = pb.measured_delta()
    # RAM (R-020 was killed at 1.74 GB): the initial basis is not used here, and the columns are
    # recomputed on the fly (the fast operator costs ~50 ms per column) instead of stored.
    pb.basis = None
    import gc; gc.collect()
    lib = colib.stateless_library(max_ypow=2)
    phi = lambda c: pb.stateless_force(*lib[c])                    # noqa: E731
    nrm = np.array([np.linalg.norm(pb.delta_of(phi(c))) for c in range(len(lib))])
    Fs = np.stack([(phi(c) @ pb.Pinv.T).reshape(-1).astype(np.float32) for c in range(len(lib))],
                  axis=1)
    Fg = (Fs.T @ Fs).astype(np.float64) / np.outer(nrm, nrm)
    del Fs; gc.collect()
    # OA-015: the size is held at OA_SIZE (||G phi||); R-020 showed that without it the least-force
    # step removes the addition (12.7 -> 0.95). Delta Gram of the normalised columns, for the size.
    size = float(os.environ.get('OA_SIZE', '0'))
    if size > 0:
        Ds32 = np.stack([pb.delta_of(phi(c)).astype(np.float32) for c in range(len(lib))], axis=1)
        Gn = (Ds32.T @ Ds32).astype(np.float64) / np.outer(nrm, nrm)
        del Ds32; gc.collect()
    # start: the given addition, expressed in library coordinates
    kappa = np.zeros(len(lib))
    for c, s in enumerate(lib):
        m, i, j, pw = s
        key = {('M', 0): 'oa_Ma0', ('M', 1): 'oa_Ma1', ('M', 2): 'oa_Ma2', ('K', 0): 'oa_Ka0', ('K', 1): 'oa_Ka1',
               ('K', 2): 'oa_Ka2'}[(m, pw)]
        kappa[c] = oa0[key][i, j]
    chk = aio.empty()
    for c, s in enumerate(lib):
        aio.add_slot(chk, s, kappa[c])
    assert all(np.allclose(chk[k], oa0[k]) for k in chk), 'start addition is not in the library'
    n_it = int(os.environ.get('OA_ITERS', '6'))
    hist = []
    for it in range(n_it + 1):
        f = sum(kappa[c] * phi(c) for c in range(len(lib)))
        bas = basis_at(pb, f)
        D = Dnull + pb.delta_of(f)                           # G is independent of u (affine step)
        t = bas.jt(bas.apply(b_null)).numpy()
        res = bas.jt(torch.from_numpy(D)).numpy() - t
        dth = bas.coefficient(torch.from_numpy(D)).numpy()
        combo = pb.blk.combinations_from_free((pb.vbar + torch.from_numpy(dth)).to(pb.device)).detach().cpu().numpy()
        combo0 = pb.blk.combinations_from_free((pb.vbar + b_null).to(pb.device)).detach().cpu().numpy()
        tc = pb.true_combo.numpy()
        scale = np.abs(tc).copy(); scale[7] = 10.45
        shift = 100 * (combo - combo0) / scale
        shift_own = 100 * (combo[7] - combo0[7]) / abs(tc[7])
        fs = f @ pb.Pinv.T
        print('[u-aware] it %d: |residual| %.3e, predicted shift vs b_null: max nine %.4f %%, m_diff own '
              '%+.4f %%, ||Delta_add|| %.4f, force rms %s N'
              % (it, np.linalg.norm(res), np.abs(np.delete(shift, 7)).max(), shift_own,
                 np.linalg.norm(D - Dnull), np.array2string(np.sqrt((fs ** 2).mean(0)), precision=2)),
              flush=True)
        hist.append(dict(it=it, residual=float(np.linalg.norm(res)),
                         shift_max_nine=float(np.abs(np.delete(shift, 7)).max()),
                         shift_mdiff_own=float(shift_own)))
        if it == n_it:
            break
        A = np.stack([bas.jt(torch.from_numpy(pb.delta_of(phi(c)))).numpy()
                      for c in range(len(lib))], axis=1) / nrm[None, :]
        del bas; gc.collect()
        n = len(lib)
        KKT = np.block([[2 * Fg, A.T], [A, np.zeros((10, 10))]])
        if size <= 0:
            z = np.linalg.solve(KKT, np.concatenate([np.zeros(n), -res]))[:n]
            kappa = kappa + z / nrm
        else:
            # affine solution set of the linearised conditions A z = A z_n - res (z = kappa * nrm):
            # least-force particular solution + the null-space part closest to the current member
            # (force metric), rescaled so that ||Delta_add|| = size.
            zn = kappa * nrm
            zp = np.linalg.solve(KKT, np.concatenate([np.zeros(n), A @ zn - res]))[:n]
            zp_null = np.linalg.solve(KKT, np.concatenate([np.zeros(n), np.zeros(10)]))[:n]
            _u, _s, Vt = np.linalg.svd(A)
            Nn = Vt[10:].T
            w = np.linalg.solve(Nn.T @ Fg @ Nn, Nn.T @ Fg @ zn)
            zh = Nn @ w
            # the floor part of the target is small; split z = z_floor + alpha * zh with
            # A z_floor = A zn - res - A zh * 0 (zh is in the null space, so A zh = 0)
            zf = zp - Nn @ np.linalg.solve(Nn.T @ Fg @ Nn, Nn.T @ Fg @ zp)   # zp minus its null part
            a2 = zh @ Gn @ zh; a1 = 2 * zf @ Gn @ zh; a0 = zf @ Gn @ zf - size ** 2
            alpha = (-a1 + np.sqrt(max(a1 * a1 - 4 * a2 * a0, 0.0))) / (2 * a2)
            kappa = (zf + alpha * zh) / nrm
    oa = aio.empty()
    for c, s in enumerate(lib):
        aio.add_slot(oa, s, kappa[c])
    aio.save_mat(os.environ['OA_OUT'], oa, 'u-aware re-solve of %s on %s'
                 % (os.path.basename(os.environ['OA_ADDITION']), pb.mode))
    json.dump(dict(history=hist, kappa={colib.slot_name(s): float(k) for s, k in zip(lib, kappa)}),
              open(os.environ['OA_OUT'].replace('.mat', '.json'), 'w'), indent=2)
    print('[out] %s' % os.environ['OA_OUT'])


if __name__ == '__main__':
    main()
