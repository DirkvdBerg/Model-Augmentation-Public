"""G4 attempt 3 (ET-011): the window of the attempt-2 encoder, chosen on the TRAIN split.

RFS (recovered-block Jacobian map + friction input correction + Y schedule) rebuilt at na in
{7, 11, 15, 21, 29}; na chosen on the 11 train records by mean_ch RMS(RFS)/RMS(P0-T); verdict by the
unchanged ET-006 rule on all 15 records and, separately, on the 4 held-out val/test records.
All encoders are scored on the SAME windows (k from the cut start + 30). A kinematic estimator is
reported as a reference arm. Output: outputs/telica/g4/g4_telica_a3.json
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g4_telica as g4                                           # noqa: E402
import g4_telica_a2 as a2                                        # noqa: E402
import core                                                      # noqa: E402
import model_map as mm                                           # noqa: E402
import tr_env                                                    # noqa: E402
import rate                                                      # noqa: E402
from pipeline.real_data import frozen_norm, PRE                  # noqa: E402
from params import telica_params as tp                           # noqa: E402
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize  # noqa: E402
from model_augmentation.utils.utils import normalize_linear_ss_matrices                      # noqa: E402
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug              # noqa: E402
from pipeline.telica_model import _synthetic_sysdata             # noqa: E402

NAS = [7, 11, 15, 21, 29]
SG_M = 11                        # HEURISTIC (ET-011): quadratic fit over the last 11 samples (2.2 ms)


def parent_na(norm, dt, na):
    Ad, Bd, Cd, Dd = gantry_linearize_and_discretize(dt=dt)
    Ab, Bb, Cb, Db = normalize_linear_ss_matrices(Ad, Bd, Cd, Dd, _synthetic_sysdata(norm))
    torch.manual_seed(42)
    return linear_encoder_init_aug(
        A=Ab, B=Bb, C=Cb, D=Db, nx=6, nu=3, ny=3, na=na, nb=na, nx_aug=8,
        n_nodes_per_layer=24, n_hidden_layers=3, flag_linear_only=False,
        u_mean=norm.u_mean, std_u=norm.std_u, y0=norm.y0, ystd=norm.ystd,
        x_mean=norm.x_mean, std_x=norm.std_x, dtype=torch.float64).to(torch.float64).eval()


def windows_at(d5, na, k):
    y, u = np.asarray(d5['q1'], float), np.asarray(d5['u'], float)
    return (np.ascontiguousarray(np.stack([y[i - na:i + 1] for i in k])),
            np.ascontiguousarray(np.stack([u[i - na:i + 1] for i in k])))


def k_common(d5):
    ks0 = max(0, int(d5['motion_idx']) - PRE // rate.D)
    return np.arange(ks0 + max(NAS) + 1, len(d5['q1']))


def kinematic(yw, dt):
    """Positions P^-T y(k); velocities P^-T x endpoint derivative of a quadratic LS fit (last SG_M)."""
    t = np.arange(-SG_M + 1, 1) * dt
    V = np.stack([np.ones_like(t), t, t ** 2], 1)
    w = np.linalg.pinv(V)[1]                                     # d/dt at t = 0
    v = np.einsum('m,bmj->bj', w, yw[:, -SG_M:, :])
    return np.concatenate([yw[:, -1, :] @ g4.PT_INV.T, v @ g4.PT_INV.T], 1)


def main():
    t0 = time.time()
    tr_env.check_no_leak()
    norm = frozen_norm(g4._Cfg)
    for k in ('x_mean', 'std_x', 'std_u', 'u_mean'):
        setattr(norm, k, np.asarray(getattr(norm, k), float))
    dt = float(rate.TS5)
    sx = norm.std_x.ravel(); su = norm.std_u.ravel(); sy = np.asarray(norm.ystd, float)
    recs = g4.load_all()
    posb, refs = [], []
    for s, op, ypos, d, d5 in recs:
        pb, bo = g4.bounds(d, g4.FC_GRID)
        posb.append(pb); refs.append(bo)
    dist = g4.pooled_distortion(refs, g4.FC_GRID)
    for bo in refs:
        for fc in g4.FC_GRID:
            bo[fc]['bound'] = np.sqrt(bo[fc]['noise'] ** 2 + dist[fc] ** 2)
    pooled_fc = {fc: np.sqrt(np.mean([bo[fc]['bound'] ** 2 for bo in refs], 0)) for fc in g4.FC_GRID}
    fc_best = [min(g4.FC_GRID, key=lambda fc: pooled_fc[fc][j]) for j in range(3)]
    blk, cc = a2.recovered_block(norm, dt)
    hfn = a2.BlockHfn(blk, norm.Cd_norm)
    v0 = float(tp.V0_TANH)
    p0 = g4.build_p0(norm, dt)

    # per record: common k, reference, bound
    R = []
    for (s, op, ypos, d, d5), bo, pb in zip(recs, refs, posb):
        k = k_common(d5)
        k20 = rate.D * k
        xref = np.concatenate([(np.asarray(d['q1'], float) @ g4.PT_INV.T)[k20],
                               np.stack([bo[fc_best[j]]['v'][k20, j] for j in range(3)], 1)], 1)
        bnd = np.concatenate([pb, [bo[fc_best[j]]['bound'][j] for j in range(3)]])
        R.append(dict(s=s, op=op, ypos=ypos, d5=d5, k=k, xref=xref, bnd=bnd))

    def ms_of(enc, na, r):
        yw, uw = windows_at(r['d5'], na, r['k'])
        return ((g4.enc_phys(enc, yw, uw, norm) - r['xref']) ** 2).mean(0)

    ms_p0 = [ms_of(p0, 29, r) for r in R]
    encs, sel = {}, {}
    for na in NAS:
        grid, off = mm.model_grid(hfn, norm, 6, na)
        W0 = grid[2]
        My = (na + 1) * 3
        par = parent_na(norm, dt, na)
        with torch.no_grad():
            par.Wb_psi_y.copy_(torch.as_tensor(W0[:, :My])); par.Wb_psi_u.copy_(torch.as_tensor(W0[:, My:]))
        tw = [windows_at(r['d5'], na, r['k']) for r in R if r['s'] == 'train']
        twf = [(yw, a2.fric_np(yw, uw, cc, v0, dt)) for yw, uw in tw]
        x_r0 = [core.numpy_sched(np.zeros((0,)), W0, yw, uf, sx, su, sy) for yw, uf in twf]
        x_gr = [core.grid_apply(grid, yw, uf, yw[:, -1, 2], sx, su, sy) for yw, uf in twf]
        gain = np.sqrt(sum(((a - b) ** 2).sum(0) for a, b in zip(x_gr, x_r0)))
        deg = None
        for dg in range(0, 11):
            C, _, _ = core.fit_schedule(grid, dg)
            xp = [core.numpy_sched(C, W0, yw, uf, sx, su, sy) for yw, uf in twf]
            ratio = np.sqrt(sum(((a - b) ** 2).sum(0) for a, b in zip(xp, x_gr))) / np.maximum(gain, 1e-300)
            if np.all(ratio <= 0.1):
                deg = dg
                break
        deg = 10 if deg is None else deg
        C, _, _ = core.fit_schedule(grid, deg)
        enc = a2.FrictionCompEncoder(core.ScheduledReconEncoder(par, C, na, 3, 3, norm.y0, norm.ystd),
                                     cc, v0, dt, norm).double().eval()
        encs[na] = (enc, deg)
        ms = [ms_of(enc, na, r) for r in R]
        tr_idx = [i for i, r in enumerate(R) if r['s'] == 'train']
        num = sum(ms[i] * len(R[i]['k']) for i in tr_idx)
        den = sum(ms_p0[i] * len(R[i]['k']) for i in tr_idx)
        sel[na] = dict(obj=float(np.mean(np.sqrt(num / den))), ratio_train=np.sqrt(num / den).tolist(),
                       deg=deg, ms=ms)
        print(f'  na {na:>2} ({(na + 1) * dt * 1e3:.1f} ms) deg {deg}: train RMS ratio to P0 '
              + ' '.join(f'{v:.3f}' for v in np.sqrt(num / den)) + f'  -> objective {sel[na]["obj"]:.4f}'
              f'  [{time.time() - t0:.0f} s]')
    na_best = min(NAS, key=lambda n: sel[n]['obj'])
    ms_b = sel[na_best]['ms']
    print(f'  chosen na = {na_best} (train objective)')

    def verdict(idx, label):
        n = sum(len(R[i]['k']) for i in idx)
        P = np.sqrt(sum(ms_p0[i] * len(R[i]['k']) for i in idx) / n)
        G = np.sqrt(sum(ms_b[i] * len(R[i]['k']) for i in idx) / n)
        B = np.sqrt(sum(R[i]['bnd'] ** 2 * len(R[i]['k']) for i in idx) / n)
        res_ = B < P
        ys = sorted({R[i]['ypos'] for i in idx})
        per_y = {}
        for y in ys:
            j = [i for i in idx if R[i]['ypos'] == y]
            m = sum(len(R[i]['k']) for i in j)
            per_y[str(y)] = (np.sqrt(sum(ms_b[i] * len(R[i]['k']) for i in j) / m)
                             / np.sqrt(sum(ms_p0[i] * len(R[i]['k']) for i in j) / m)).tolist()
        worst = max(max(np.asarray(v)[res_]) for v in per_y.values())
        lower = bool(np.all(G[res_] < P[res_]))
        ok = bool(res_.any() and lower and worst <= 1.01)
        print(f'\n  [{label}] {len(idx)} records; bound ' + ' '.join(f'{v:.2e}' for v in B))
        print('    P0-T ' + ' '.join(f'{v:.3e}' for v in P))
        print('    RFS  ' + ' '.join(f'{v:.3e}' for v in G) + '   ratio ' + ' '.join(f'{v:.3f}' for v in G / P))
        for y, v in per_y.items():
            print(f'    ypos {float(y):>6.0f}  ' + ' '.join(f'{c} {x:.3f}' for c, x in zip(g4.CH, v)))
        print(f'    verdict: lower on all resolvable {lower}; worst ypos ratio {worst:.3f} -> {"PASS" if ok else "FAIL"}')
        return dict(P0=P.tolist(), RFS=G.tolist(), bound=B.tolist(), resolvable=res_.tolist(),
                    per_ypos=per_y, worst=float(worst), lower=lower, PASS=ok)

    all_idx = list(range(len(R)))
    held = [i for i, r in enumerate(R) if r['s'] != 'train']
    out = dict(na_best=na_best, selection={str(k): {kk: vv for kk, vv in v.items() if kk != 'ms'} for k, v in sel.items()},
               all15=verdict(all_idx, 'all 15 records (ET-006 rule)'),
               heldout=verdict(held, 'held-out val + test'))
    # kinematic reference arm
    kin = np.zeros(6); n = 0
    for r in R:
        yw, _ = windows_at(r['d5'], SG_M, r['k'])
        kin += ((kinematic(yw, dt) - r['xref']) ** 2).sum(0); n += len(r['k'])
    out['kinematic'] = np.sqrt(kin / n).tolist()
    print('\n  kinematic reference arm (not a candidate), pooled RMS: '
          + ' '.join(f'{c} {v:.2e}' for c, v in zip(g4.CH, out['kinematic'])))
    with open(os.path.join(g4.OUT, 'g4_telica_a3.json'), 'w') as f:
        json.dump(out, f, indent=1, default=float)
    enc, deg = encs[na_best]
    torch.save(dict(na=na_best, deg=deg, rfs_state=enc.state_dict()), os.path.join(g4.OUT, 'g2_telica_a3_encoder.pt'))
    print(f'  saved outputs/telica/g4/g4_telica_a3.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
