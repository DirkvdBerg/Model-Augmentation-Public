"""Frozen basis versus per-epoch re-anchoring, with drift logging (D-183).

Category: DEVELOPMENT AND FALSIFICATION TOOL. Not gantry evidence.

WHY THIS EXISTS. Gyorok precomputes the projection basis at `theta_0` and argues the
approximation holds because `theta_0` is physically derived and close to the true system. That
premise fails for us: under joint estimation the anchor is a deliberate 10% detune, and 11
degrees of subspace rotation from that detune alone is already on record. The stated reason not
to re-anchor was a ~6 min rebuild, which was measured to be 0.45 s once the Jacobian is taken in
forward mode. So the cost argument is gone and only the methodological one remains: a moving
basis makes the objective non-stationary and lets the network chase a constraint set that moves
under the prediction loss.

This run measures which of those actually happens.

THE GUARD (D-182), and it is why this is not a drop-in change. `V = beta ||Q^T d||^2` can be
reduced two ways: change the network so its contribution leaves the subspace, which is the
point, or MOVE the subspace via `theta`, which costs nothing in prediction error and is usually
cheaper. Freezing made the second impossible for free. Re-anchoring does not, so every rebuild
happens under `no_grad`, the result is installed as a constant, and
`assert_no_gradient_path` is called after each one rather than only at startup.

WHAT DRIFT ANGLES MEAN. Decaying toward zero: `theta` is settling and the frozen approximation
was defensible. Persisting or oscillating: the network is chasing a moving target, which is the
instability this algorithm risks and the reason it is kept as a separate method rather than
replacing the frozen one.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/reanchor_compare.py
"""
__project_origin__ = "added"

import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import system as S            # noqa: E402
import geometry as G          # noqa: E402
import train_compare as T     # noqa: E402
from testbed_adapter import TestbedAdapter                       # noqa: E402
from model_augmentation.fit_systems.trajectory_adapter import build_geometry  # noqa: E402
from model_augmentation.fit_systems.trajectory_orth_projection import (  # noqa: E402
    subspace_drift, assert_no_gradient_path)

F64 = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'reanchor_compare.json')

N_WIN = 24
N_ITER = 600
ITERS_PER_EPOCH = 50          # 12 rebuilds over the run
LR = 2e-2
BETA = 1.0


def setup(seed=0):
    d_tr, d_va = S.make_dataset(seed=0), S.make_dataset(seed=7)
    rng = np.random.default_rng(seed)
    t_win = T.T_WIN
    st = rng.choice(np.arange(T.N_LAG, len(d_tr['u']) - t_win - 1), N_WIN, replace=False)
    sv = rng.choice(np.arange(T.N_LAG, len(d_va['u']) - t_win - 1), N_WIN, replace=False)
    U, Y, L = T.make_batch(d_tr['u'], d_tr['y'], st)
    Uv, Yv, Lv = T.make_batch(d_va['u'], d_va['y'], sv)
    _, shapes = G.init_eta(0)
    return dict(d_tr=d_tr, d_va=d_va, st=st, sv=sv, U=U, Y=Y, L=L,
                Uv=Uv, Yv=Yv, Lv=Lv, shapes=shapes, t_win=t_win)


def run(mode, ctx, verbose=True):
    """mode: 'frozen' | 'reanchor'. Joint estimation on, extension penalty on d_total."""
    shapes = ctx['shapes']
    th0 = torch.tensor(S.THETA_TRUE * T.DETUNE, dtype=F64)
    p_th = torch.zeros(4, dtype=F64, requires_grad=True)
    eta0, _ = G.init_eta(0)
    eta = eta0.clone().requires_grad_(True)
    xi = torch.zeros(T.NX * 2 * T.N_LAG + T.NX, dtype=F64)
    xi[:T.NX * 2 * T.N_LAG] = torch.linalg.lstsq(
        ctx['L'], torch.tensor(S.baseline_states(ctx['d_tr']['u'])[ctx['st']],
                               dtype=F64)).solution.T.reshape(-1)
    xi = xi.requires_grad_(True)

    ad = TestbedAdapter(ctx['U'], ctx['L'], shapes, N_WIN, ctx['t_win'])
    opt = torch.optim.Adam([p_th, eta, xi], lr=LR)

    def rebuild(theta_now):
        """D-182: OUTSIDE the graph, installed as a constant, checked at every rebuild."""
        with torch.no_grad():
            params = dict(physical=theta_now.detach().clone(),
                          augmentation=eta.detach().clone(),
                          encoder=xi.detach().clone())
        g = build_geometry(ad, params, verbose=False)
        assert_no_gradient_path(g)
        return g

    geom = rebuild(th0 * torch.exp(p_th.detach()))
    # RETAIN the actual starting geometry. BUG FIXED 2026-09-07 (found in review): the
    # start-to-finish drift was computed as `rebuild(th0)` at the END, but `rebuild` closes over
    # the CURRENT augmentation and encoder, which by then are the final ones. It therefore
    # compared initial theta + FINAL network against final theta + final network, which is why
    # both arms reported exactly 2.23 deg.
    geom_initial = geom
    drift, kap_t = [], np.array([S.M1, S.C1, S.KA_TRUE + S.KB_TRUE])

    for it in range(N_ITER):
        if mode == 'reanchor' and it > 0 and it % ITERS_PER_EPOCH == 0:
            new = rebuild(th0 * torch.exp(p_th.detach()))
            dm = subspace_drift(geom.Q_S, new.Q_S)
            dm['iter'] = it
            drift.append(dm)
            geom = new
            if verbose:
                print(f'    rebuild @ {it:4d}: max angle {dm["max_deg"]:6.2f} deg  '
                      f'mean {dm["mean_deg"]:6.2f}  rank {dm["rank_prev"]}->{dm["rank_new"]}',
                      flush=True)
        opt.zero_grad()
        theta = th0 * torch.exp(p_th)
        yh = T.rollout(theta, eta, xi, ctx['U'], ctx['L'], shapes, mode='full')
        loss = ((yh - ctx['Y']) ** 2).mean()
        # Penalty rollout with theta and the encoder as CONSTANTS: the penalty must not move
        # them (D-182). The prediction loss above still trains them normally.
        tc, xc = theta.detach(), xi.detach()
        dt = (T.rollout(tc, eta, xc, ctx['U'], ctx['L'], shapes, mode='full').reshape(-1)
              - T.rollout(tc, eta, xc, ctx['U'], ctx['L'], shapes, mode='annoff').reshape(-1))
        loss = loss + geom.penalty(dt, BETA)
        loss.backward()
        opt.step()

    with torch.no_grad():
        th = (th0 * torch.exp(p_th)).detach()
        kap = np.array([th[0].item(), th[1].item(), (th[2] + th[3]).item()])
        kap_err = float(np.linalg.norm((kap - kap_t) / kap_t))
        yhv = T.rollout(th, eta.detach(), xi.detach(), ctx['Uv'], ctx['Lv'], shapes, 'full')
        rmse = float(((yhv - ctx['Yv']) ** 2).mean().sqrt())
        full = T.rollout(th, eta.detach(), xi.detach(), ctx['U'], ctx['L'], shapes,
                         'full').reshape(-1)
        off = T.rollout(th, eta.detach(), xi.detach(), ctx['U'], ctx['L'], shapes,
                        'annoff').reshape(-1)
        m = geom.contribution_metrics(full - off)

    # How far did the FINAL geometry move from the one training ACTUALLY started with?
    g_end = rebuild(th)
    total = subspace_drift(geom_initial.Q_S, g_end.Q_S)
    return dict(mode=mode, kappa_err=kap_err, val_rmse=rmse, ell=m['ell'],
                a_par=m['a_par'], a_perp=m['a_perp'], drift=drift,
                total_drift_deg=total['max_deg'], kappa=kap.tolist())


def main():
    torch.set_default_dtype(F64)
    t0 = time.time()
    print('=' * 78)
    print('FROZEN vs PER-EPOCH RE-ANCHORING (D-183).  Development tool, not gantry evidence.')
    print('=' * 78)
    ctx = setup()
    print(f'  joint estimation ON, theta detuned {[f"{v:+.0%}" for v in (T.DETUNE - 1)]}, '
          f'beta={BETA}')
    print(f'  {N_ITER} iters, rebuild every {ITERS_PER_EPOCH} '
          f'({N_ITER // ITERS_PER_EPOCH - 1} rebuilds), {N_WIN} windows\n')

    out = {}
    for mode in ('frozen', 'reanchor'):
        print(f'  --- {mode} ---')
        r = run(mode, ctx)
        out[mode] = r
        print(f'    kappa_err {r["kappa_err"]:.4f}   val_rmse {r["val_rmse"]:.3e}   '
              f'ell {r["ell"]:.4f}   a_perp {r["a_perp"]:.3e}')
        print(f'    basis rotation from start to finish: {r["total_drift_deg"]:.2f} deg\n')

    print('-' * 78)
    f, r = out['frozen'], out['reanchor']
    print(f'  kappa error : frozen {f["kappa_err"]:.4f}   reanchor {r["kappa_err"]:.4f}   '
          f'({r["kappa_err"] / f["kappa_err"]:.2f}x)')
    print(f'  val rmse    : frozen {f["val_rmse"]:.3e}  reanchor {r["val_rmse"]:.3e}')
    print(f'  ell         : frozen {f["ell"]:.4f}   reanchor {r["ell"]:.4f}')
    if r['drift']:
        seq = [d['max_deg'] for d in r['drift']]
        print(f'\n  per-rebuild max angle (deg): '
              + ' '.join(f'{v:.2f}' for v in seq))
        print('  Decaying toward zero means theta is settling. NOTE: small CONSECUTIVE angles')
        print('  do not bound ACCUMULATED drift; read them with the start-to-finish figure.')
        print('  Persisting or oscillating means the network is chasing a moving constraint set.')
        trend = 'DECAYING' if seq[-1] < 0.5 * seq[0] else (
            'FLAT/OSCILLATING' if seq[-1] < 2 * seq[0] else 'GROWING')
        print(f'  observed: {trend}  (first {seq[0]:.2f}, last {seq[-1]:.2f})')
    print(f'\n  wall clock {time.time() - t0:.1f} s')
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f'  wrote {OUT}')


if __name__ == '__main__':
    main()
