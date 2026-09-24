"""Gantry OBC diagnostics (D-192): dataset qualification, the absorber truth discrepancy, `rho`.

Bounded PREFLIGHT tools, run by `orthogonal-by-construction/implementation/08-one-step/`; nothing
here is on the training path. Three groups:

* `qualify_basis`            all ten singular values, numerical rank under the declared tolerance
                             and a sweep of tolerances, condition number, normalized column norms
                             and pairwise correlations, per-record contribution to each column
                             norm, and leave-one-record-out rank and smallest singular value.
* `absorber_truth_one_step`  the 8-state truth plant of the augmentation data
                             (`Matlab-scripts/Augmentation/gantrySystemExtended.m`, ported in
                             `scripts/gantry/common/oracle.py`) with the absorber parameters as
                             ARGUMENTS, vectorised over reference tuples, so the one-step
                             discrepancy `delta` can be formed at recorded and at zero absorber
                             states and the parameter hypothesis checked against the data.
* `rho_table`                `rho(B; delta) = ||B B^+ delta|| / ||delta||` for `B` in
                             `{J, [J | Gamma]}` crossed with the two discrepancies.
"""
__project_origin__ = "added"

import time
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from model_augmentation.systems.gantry_ss import (
    m1 as _m1, m2 as _m2, mb as _mb, mh as _mh, Jb as _Jb, Jh as _Jh,
    cg1 as _cg1, cg2 as _cg2, cy as _cy, cb1 as _cb1, cb2 as _cb2,
    kb1 as _kb1, kb2 as _kb2, Lb as _Lb, d as _d, P as _P)
from model_augmentation.fit_systems.obc import OBCBasis

from .obc_gantry import GantryReferenceSet, COMBO_NAMES


# ---------------------------------------------------------------------- qualification
def qualify_basis(J: Tensor, ref: GantryReferenceSet, row_ix: Sequence[int],
                  rtol: Optional[float] = None, names: Sequence[str] = COMBO_NAMES,
                  rtol_sweep=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14),
                  leave_one_out: bool = True, verbose: bool = True) -> dict:
    """Everything section 9.4 of the handoff asks for, from a float64 stack `J`."""
    t0 = time.perf_counter()
    J = J.to(torch.float64)
    n_rows, p = J.shape
    nr = len(row_ix)
    S = torch.linalg.svdvals(J)
    tol = OBCBasis.rank_tolerance(S, n_rows, p, rtol)
    rank = int((S > tol).sum())
    out = {
        'n_rows': n_rows, 'n_cols': p, 'n_tuples': ref.n, 'rows_per_tuple': nr,
        'singular_values': S.tolist(), 'rank': rank, 'tol': tol, 'rtol': tol / float(S[0]),
        'cond': float(S[0] / S[rank - 1]) if rank > 0 else float('inf'),
        'rank_sweep': {str(r): int((S > r * S[0]).sum()) for r in rtol_sweep},
    }
    norms = torch.linalg.vector_norm(J, dim=0)
    Jn = J / norms
    corr = Jn.T @ Jn
    out['col_norms'] = norms.tolist()
    out['col_corr'] = corr.tolist()
    # singular values of the column-NORMALIZED matrix: conditioning without the scale disparity
    Sn = torch.linalg.svdvals(Jn)
    out['singular_values_colnorm'] = Sn.tolist()
    out['cond_colnorm'] = float(Sn[0] / Sn[-1])
    # per-record contribution to each column norm (fraction of ||J[:, j]||^2)
    rec = ref.record_ix.repeat_interleave(nr).to(J.device)
    n_rec = len(ref.n_per_record)
    contrib = torch.zeros(n_rec, p, dtype=torch.float64, device=J.device)
    contrib.index_add_(0, rec, J ** 2)
    contrib = contrib / (norms ** 2)
    out['record_contribution'] = contrib.tolist()
    out['record_names'] = list(ref.record_names[:n_rec])
    # leave-one-record-out
    if leave_one_out:
        loo = []
        for r in range(n_rec):
            keep = rec != r
            Sr = torch.linalg.svdvals(J[keep])
            tol_r = OBCBasis.rank_tolerance(Sr, int(keep.sum()), p, rtol)
            loo.append({'record': ref.record_names[r], 'rank': int((Sr > tol_r).sum()),
                        'sigma_min': float(Sr[-1]), 'sigma_min_rel': float(Sr[-1] / Sr[0]),
                        'cond': float(Sr[0] / Sr[-1])})
        out['leave_one_out'] = loo
    Y = ref.x_phys[:, 2].numpy()
    out['Y_coverage'] = {'min': float(Y.min()), 'max': float(Y.max()), 'mean': float(Y.mean()),
                         'std': float(Y.std()),
                         'hist_edges': np.linspace(-0.32, 0.32, 17).tolist(),
                         'hist': np.histogram(Y, np.linspace(-0.32, 0.32, 17))[0].tolist()}
    out['seconds'] = time.perf_counter() - t0
    if verbose:
        print_qualification(out, names)
    return out


def print_qualification(q: dict, names: Sequence[str] = COMBO_NAMES) -> None:
    print('  J: %d x %d (%d tuples x %d rows); rank %d/%d at tol %.3e (rtol %.3e); cond %.4e; '
          'column-normalized cond %.4e' % (q['n_rows'], q['n_cols'], q['n_tuples'],
                                           q['rows_per_tuple'], q['rank'], q['n_cols'],
                                           q['tol'], q['rtol'], q['cond'], q['cond_colnorm']))
    print('  singular values:            ' + ' '.join('%.4e' % s for s in q['singular_values']))
    print('  singular values (col-norm): ' + ' '.join('%.4e' % s for s in q['singular_values_colnorm']))
    print('  rank vs rtol: ' + ', '.join('%s -> %d' % kv for kv in q['rank_sweep'].items()))
    print('  column norms: ' + ' '.join('%s=%.3e' % (n, v) for n, v in zip(names, q['col_norms'])))
    C = q['col_corr']
    pairs = sorted(((abs(C[i][j]), names[i], names[j], C[i][j]) for i in range(len(names))
                    for j in range(i + 1, len(names))), reverse=True)
    print('  |correlations| above 0.3: ' + ', '.join('%s~%s=%+.4f' % (a, b, c)
                                                     for m, a, b, c in pairs if m > 0.3))
    print('  per-record share of each column norm^2 (rows: record, cols: %s)' % ' '.join(names))
    for rn, row in zip(q['record_names'], q['record_contribution']):
        print('    %-24s ' % rn + ' '.join('%6.3f' % v for v in row))
    if 'leave_one_out' in q:
        print('  leave-one-record-out: rank, sigma_min, sigma_min/sigma_0, cond')
        for e in q['leave_one_out']:
            print('    %-24s rank %2d  sigma_min %.4e  rel %.3e  cond %.3e'
                  % (e['record'], e['rank'], e['sigma_min'], e['sigma_min_rel'], e['cond']))
    yc = q['Y_coverage']
    print('  Y coverage: [%+.3f, %+.3f] mean %+.4f std %.4f; histogram %s'
          % (yc['min'], yc['max'], yc['mean'], yc['std'], yc['hist']))


def principal_angles(A: Tensor, B: Tensor) -> Tensor:
    """Principal angles (rad) between range(A) and range(B), via thin QR."""
    Qa, _ = torch.linalg.qr(A.to(torch.float64))
    Qb, _ = torch.linalg.qr(B.to(torch.float64))
    s = torch.linalg.svdvals(Qa.T @ Qb).clamp(-1, 1)
    return torch.arccos(s)


# ---------------------------------------------------------------------- truth plant
def absorber_truth_one_step(x6: np.ndarray, x_abs: np.ndarray, u_stage: np.ndarray, ts: float,
                            up_sample: int, ma_frac: float, zeta_a: float,
                            fa: float = 150.0, L0: float = 0.10):
    """One RK4 step of the 8-state truth plant at every row. Returns (x6_next, x_abs_next).

    Port of `scripts/gantry/common/oracle.py::_M/_deriv/simulate` (itself an RK4 port of
    `gantrySystemExtended.m`) with the absorber parameters as ARGUMENTS and vectorised over N:
    `ma = ma_frac * mh`, `mh_rigid = mh - ma`, `ka = ma (2 pi fa)^2` (THEORY: k = m w^2),
    `ca = 2 zeta_a sqrt(ka ma)` (THEORY: c = 2 zeta sqrt(k m)), gtd_config.m:52-71.
    State per row `[X, Theta, Y, delta_a, dX, dTheta, dY, vdelta_a]`, logical forces
    `u_log = P u_stage`.
    """
    m1, m2, mb, mh = float(_m1), float(_m2), float(_mb), float(_mh)
    Jb, Jh = float(_Jb), float(_Jh)
    cg1, cg2, cy = float(_cg1), float(_cg2), float(_cy)
    cb1, cb2, kb1, kb2 = float(_cb1), float(_cb2), float(_kb1), float(_kb2)
    Lb, d = float(_Lb), float(_d)
    P_np = _P.detach().cpu().numpy().astype(np.float64)
    ma = ma_frac * mh
    mhr = mh - ma
    ka = ma * (2 * np.pi * fa) ** 2
    ca = 2 * zeta_a * np.sqrt(ka * ma)
    C4 = np.array([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0, 0],
                   [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4, 0, 0],
                   [0, 0, cy, 0], [0, 0, 0, ca]])
    K4 = np.array([[0, 0, 0, 0], [0, kb1 + kb2, 0, 0], [0, 0, 0, 0], [0, 0, 0, ka]])
    E43 = np.vstack([np.eye(3), np.zeros((1, 3))])

    def M(Y, da):
        N = Y.shape[0]
        off = (m1 - m2) * Lb / 2 - (mhr + ma) * Y - ma * L0 - ma * da
        Mm = np.zeros((N, 4, 4))
        Mm[:, 0, 0] = m1 + m2 + mb + mhr + ma
        Mm[:, 0, 1] = off; Mm[:, 1, 0] = off
        Mm[:, 1, 1] = (Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + (mhr + ma) * d ** 2
                       + mhr * Y ** 2 + ma * (Y + L0 + da) ** 2)
        Mm[:, 1, 2] = -(mhr + ma) * d; Mm[:, 2, 1] = -(mhr + ma) * d
        Mm[:, 1, 3] = -ma * d; Mm[:, 3, 1] = -ma * d
        Mm[:, 2, 2] = mhr + ma; Mm[:, 2, 3] = ma; Mm[:, 3, 2] = ma; Mm[:, 3, 3] = ma
        return Mm

    def deriv(x, u_log):
        q, qd = x[:, :4], x[:, 4:]
        rhs = u_log @ E43.T - q @ K4.T - qd @ C4.T
        qdd = np.linalg.solve(M(x[:, 2], x[:, 3]), rhs[..., None])[..., 0]
        return np.concatenate([qd, qdd], 1)

    x = np.concatenate([x6[:, :3], x_abs[:, :1], x6[:, 3:], x_abs[:, 1:]], 1).astype(np.float64)
    u_log = u_stage.astype(np.float64) @ P_np.T
    h = ts / up_sample
    for _ in range(up_sample):
        k1 = deriv(x, u_log)
        k2 = deriv(x + 0.5 * h * k1, u_log)
        k3 = deriv(x + 0.5 * h * k2, u_log)
        k4 = deriv(x + h * k3, u_log)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    x6n = np.concatenate([x[:, :3], x[:, 4:7]], 1)
    xan = np.stack([x[:, 3], x[:, 7]], 1)
    return x6n, xan


def check_truth_parameters(ref: GantryReferenceSet, up_sample: int, hypotheses) -> list:
    """One-step prediction error of the truth plant from the RECORDED state, per hypothesis.

    Uses the recorded `x_logical` at k and k+1 where both exist, so the number reported is the
    integration error plus any parameter mismatch. The hypothesis with the smallest error is the
    data-derived choice for `delta`; a hypothesis that is wrong shows an error orders above it.
    """
    # x_phys is the FD reference state; for this check use the RECORDED state at k: rebuild from
    # x_phys_next of the previous sample is not available, so use the reference tuples whose
    # velocity is the FD estimate and report the next-state error relative to its scale.
    out = []
    xk = ref.x_phys.numpy(); xa = ref.x_abs.numpy(); u = ref.u_stage.numpy()
    xn_rec = ref.x_phys_next.numpy()
    scale = xn_rec.std(0)
    for name, ma_frac, zeta in hypotheses:
        xn, _ = absorber_truth_one_step(xk, xa, u, ref.ts, up_sample, ma_frac, zeta)
        err = np.sqrt(((xn - xn_rec) ** 2).mean(0)) / scale
        out.append({'name': name, 'ma_frac': ma_frac, 'zeta_a': zeta,
                    'rel_rms_per_row': err.tolist(), 'rel_rms': float(np.sqrt((err ** 2).mean()))})
    return out


# ---------------------------------------------------------------------- discrepancy and rho
def discrepancy_fields(ref: GantryReferenceSet, base_next_n: Tensor, norm, up_sample: int,
                       ma_frac: float, zeta_a: float, row_ix: Sequence[int]) -> dict:
    """The one-step physical discrepancy in NORMALIZED coordinates on the routed rows, stacked
    sample-major like `J`, for three definitions:

      'recorded'  truth step at the reference tuple with the RECORDED absorber state
      'zero'      truth step at the reference tuple with ZERO absorber state (the x_a = 0 slice)
      'data'      recorded next logical state minus the baseline step (fully data-derived)

    `base_next_n` is the baseline transition at `vbar` on the reference tuples, normalized,
    `(N, nx_phys)`.
    """
    x_mean = np.asarray(norm.x_mean, dtype=np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, dtype=np.float64).reshape(1, 6)
    xk = ref.x_phys.numpy(); xa = ref.x_abs.numpy(); u = ref.u_stage.numpy()
    base = base_next_n.detach().cpu().to(torch.float64).numpy()
    rows = list(row_ix)
    out = {}
    xn_rec, _ = absorber_truth_one_step(xk, xa, u, ref.ts, up_sample, ma_frac, zeta_a)
    xn_zero, _ = absorber_truth_one_step(xk, np.zeros_like(xa), u, ref.ts, up_sample, ma_frac, zeta_a)
    for key, xn in (('recorded', xn_rec), ('zero', xn_zero), ('data', ref.x_phys_next.numpy())):
        dn = (xn - x_mean) / std_x - base
        out[key] = torch.from_numpy(np.ascontiguousarray(dn[:, rows]).reshape(-1))
    return out


def rho_table(J: Tensor, Gamma: Tensor, deltas: dict, eps: float = 1e-300) -> dict:
    """`rho(B; delta)` for `B in {J, [J | Gamma]}` and every delta. Also the rank of `[J|Gamma]`
    and the fraction of `Gamma` outside `range(J)`."""
    J = J.to(torch.float64)
    G = Gamma.to(torch.float64).reshape(-1, 1)
    UJ, SJ, _ = torch.linalg.svd(J, full_matrices=False)
    JG = torch.cat([J, G], 1)
    UE, SE, _ = torch.linalg.svd(JG, full_matrices=False)
    tolE = OBCBasis.rank_tolerance(SE, JG.shape[0], JG.shape[1])
    rankE = int((SE > tolE).sum())
    out = {'rank_J': int((SJ > OBCBasis.rank_tolerance(SJ, J.shape[0], J.shape[1])).sum()),
           'rank_JG': rankE, 'sv_JG': SE.tolist(),
           'gamma_outside_J': float(torch.linalg.vector_norm(G[:, 0] - UJ @ (UJ.T @ G[:, 0]))
                                    / (torch.linalg.vector_norm(G[:, 0]) + eps)),
           'rho': {}}
    for key, dlt in deltas.items():
        dlt = dlt.to(J.device, torch.float64)
        nd = torch.linalg.vector_norm(dlt) + eps
        out['rho'][key] = {
            'J': float(torch.linalg.vector_norm(UJ @ (UJ.T @ dlt)) / nd),
            'J|Gamma': float(torch.linalg.vector_norm(UE[:, :rankE] @ (UE[:, :rankE].T @ dlt)) / nd),
            'delta_norm': float(nd),
        }
    return out
