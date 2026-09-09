"""Four-condition joint-training comparison on the testbed, with TRUE parameter error.

Category: DEVELOPMENT AND FALSIFICATION TOOL (see system.py). Not gantry evidence.

This is the experiment the gantry cannot give us cheaply: `theta_true` is known, so parameter
error is measured directly instead of inferred from geometry.

CONDITIONS

  0  baseline only          no augmentation at all. THE REFERENCE.
  1  joint, no penalty      augmentation free. Negation is possible and unconstrained.
  2  joint + pointwise      the Gyorok analogue: Q from the one-step parameter Jacobian at
                            points with x_aug = 0, penalty beta ||Q^T f_ann||^2.
  5  joint + both           extension AND pointwise together, same beta on each. They are
                            not substitutes: d_total is a rollout quantity and only ever
                            sees the REALISED trajectory, so it fixes COMPOSITION; the
                            pointwise term constrains the network at sampled points OFF
                            that trajectory, so it fixes COVERAGE. This condition measures
                            whether the coverage term adds anything once composition is
                            handled.
  4  joint + size penalty   THE CONTROL: beta ||d||^2, an ordinary contribution-size
                            penalty with no projection at all. Without it, "the extension
                            improves parameters" cannot be distinguished from "suppressing
                            the augmentation at all improves parameters". Note that equal
                            beta grids do NOT imply comparable regularisation strength
                            across differently scaled penalties, so compare the
                            prediction-versus-parameter-error TRADEOFF, not single points.
  3  joint + extension      penalty on the TOTAL windowed contribution to the stacked output,
                            beta ||Q_S^T M_D d||^2, with d the difference between the full
                            rollout and the same rollout with the ANN output zeroed. This
                            covers BOTH the direct correction and the additional-state path;
                            `d_state` is retained as a diagnostic, not as the penalised term.

THE FALSIFIABLE PREDICTION, from gaps.md Section 0:

  > Orthogonality leaves the parameter estimate no worse than the baseline-alone fit.

  So condition 3 should land at parameter error close to condition 0, while condition 1 is
  indeterminate and may be much better OR much worse. If condition 3 comes out systematically
  worse than condition 0, Section 0's framing is wrong and must be retracted.

Parameter error is reported in IDENTIFIABLE coordinates `kappa = (m, c, ka + kb)`. Reporting it
in the raw four would be meaningless: the `(0,0,1,-1)` direction is exactly unidentifiable, so
any split of `ka` and `kb` is equally valid and its "error" is an artefact.

Both penalised conditions get the SAME beta grid, and the frontier is reported rather than a
single tuned point, so neither method can be flattered by tuning the other badly.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/testbed/train_compare.py
"""
__project_origin__ = "added"

import os
import sys
import json
import time

import numpy as np
import torch
from torch.func import jacfwd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system as S            # noqa: E402
import geometry as G          # noqa: E402

F64 = torch.float64
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'train_compare.json')

NX, NA, NU = G.NX, G.NA, G.NU
N_WIN, N_LAG = G.N_WIN, G.N_LAG
T_WIN = 32            # 0.64 s at 50 Hz. Measured 2026-09-06: T=32 converges BETTER than
                      # T=50 (kappa_err 0.073 vs 0.164 at 300 iters) as well as faster, because
                      # shorter BPTT windows are better conditioned. Steps drive cost; windows
                      # are batched and nearly free, so N_WIN stays high.
N_ITER = 600          # lr is at its stability limit (lr=0.15 diverges to kappa_err 29,
                      # lr=0.06 is worse than 0.02), so iterations cannot be traded for lr.
LR = 2e-2
# theta is trained in LOG coordinates: theta = theta_init * exp(p), p initialised at 0.
# Without this a single Adam learning rate cannot serve parameters of different magnitude
# (k ~ 100, m ~ 1): steps are ~lr per coordinate, so k moves far too slowly while m is
# destroyed. The first run of this script did exactly that, giving a baseline-only kappa
# error of 2.17 against a 0.17 starting point. The deployed gantry block uses `log_params`
# for the same reason.
DETUNE = np.array([1.10, 0.90, 1.10, 1.10])     # 10% off truth, as in the gantry runs
BETAS = [1e-2, 1e0, 1e2]
N_PTS = 400                                      # pointwise-penalty evaluation points


# ------------------------------------------------------------------ batched predictor
def make_batch(u, y, starts):
    U = torch.tensor(np.stack([u[k0:k0 + T_WIN] for k0 in starts]), dtype=F64)
    Y = torch.tensor(np.stack([y[k0:k0 + T_WIN] for k0 in starts]), dtype=F64)
    L = torch.tensor(np.stack([np.concatenate([u[k0 - N_LAG:k0], y[k0 - N_LAG:k0]])
                               for k0 in starts]), dtype=F64)
    return U, Y, L


def ann_b(Z, W):
    """Batched MLP from PRE-UNPACKED weights. Z is (B, n_in).

    Physical-correction rows and latent rows carry DIFFERENT scales; see geometry.ann.
    A single shared scale put the additional-state path at O(eps^2) and the direct path at
    O(eps), which is why the state path carried only 2.5e-4 of the contribution.
    """
    W1, b1, W2, b2, W3, b3 = W
    h = torch.tanh(Z @ W1.T + b1)
    h = torch.tanh(h @ W2.T + b2)
    w = h @ W3.T + b3
    return torch.cat([G.SCALE_PHYS * w[:, :NX], G.SCALE_LAT * w[:, NX:]], dim=1)


def rollout(theta, eta, xi, U, L, shapes, mode='full'):
    """Batched rollout. mode: 'full' | 'frozen' (x_aug := 0) | 'annoff'. Returns (B, T).

    `unpack` and `baseline_dt` are HOISTED OUT of the step loop. Profiling (2026-09-06) showed
    unpack inside the loop cost 0.34 ms x T = 17 ms of a 50 ms forward pass, a third of the
    runtime, for work that changes only once per iteration.
    """
    Ad, Bd = S.baseline_dt(theta)
    AdT, BdT = Ad.T, Bd.T
    W = None if mode == 'annoff' else G.unpack(eta, shapes)
    Wenc = xi[:NX * 2 * N_LAG].reshape(NX, 2 * N_LAG)
    benc = xi[NX * 2 * N_LAG:]
    x = L @ Wenc.T + benc
    B = U.shape[0]
    xa = torch.zeros(B, NA, dtype=F64)
    zero_w = torch.zeros(B, NX + NA, dtype=F64)
    ys = []
    for k in range(U.shape[1]):
        ys.append(x[:, 0])
        uk = U[:, k:k + 1]
        w = zero_w if mode == 'annoff' else ann_b(torch.cat([x, xa, uk], dim=1), W)
        x = x @ AdT + uk @ BdT + w[:, :NX]
        xa = torch.zeros_like(xa) if mode == 'frozen' else w[:, NX:]
    return torch.stack(ys, dim=1)


# ------------------------------------------------------------------ penalty bases (frozen)
def pointwise_basis(theta0, u, y):
    """Gyorok analogue: Q from the one-step parameter Jacobian at points with x_aug = 0."""
    _, Xb = S.simulate_baseline(u, theta0.detach().numpy())
    rng = np.random.default_rng(3)
    ix = rng.choice(len(u) - 1, size=N_PTS, replace=False)
    # One batched forward-mode Jacobian over all points. The first version rebuilt the matrix
    # exponential inside the loop, once per point, for a quantity that does not depend on i.
    Xp = torch.tensor(np.stack([Xb[i] for i in ix]), dtype=F64)
    Up = torch.tensor(np.stack([u[i:i + 1] for i in ix]), dtype=F64)

    def onestep(th):
        Ad, Bd = S.baseline_dt(th)
        return (Xp @ Ad.T + Up @ Bd.T).reshape(-1)

    Phi = jacfwd(onestep)(theta0).detach().numpy()        # (N_PTS*NX, 4)
    Q, s, _ = np.linalg.svd(Phi, full_matrices=False)
    r = int((s > 1e-12 * s[0]).sum())
    Z = torch.tensor(np.stack([np.concatenate([Xb[i], np.zeros(NA), u[i:i + 1]])
                               for i in ix]), dtype=F64)
    return torch.tensor(Q[:, :r], dtype=F64), Z, r


def extension_basis(theta0, eta0, xi0, U, L, shapes):
    """Q_S from the windowed OUTPUT sensitivity, encoder profiled out. Frozen and detached.

    FORWARD mode, not reverse. The stacked output has N_WIN*T_WIN entries (order 1500) while
    the inputs are 4 and 34 parameters, so reverse mode costs ~1500 passes and forward mode
    costs 38. The first version used `torch.autograd.functional.jacobian` (reverse) and this
    single line was ~85% of the whole sweep's runtime: 610 s of 731 s.
    """
    f = lambda th, x_: rollout(th, eta0, x_, U, L, shapes).reshape(-1)
    Smat = jacfwd(lambda th: f(th, xi0))(theta0).detach().numpy()
    Emat = jacfwd(lambda x_: f(theta0, x_))(xi0).detach().numpy()
    QE, sE, _ = np.linalg.svd(Emat, full_matrices=False)
    rE = int((sE > 1e-12 * sE[0]).sum())
    MD = np.eye(Smat.shape[0]) - QE[:, :rE] @ QE[:, :rE].T
    QS, s, _ = np.linalg.svd(MD @ Smat, full_matrices=False)
    r = int((s > 1e-12 * s[0]).sum())
    return (torch.tensor(QS[:, :r], dtype=F64), torch.tensor(MD, dtype=F64), r)


# ------------------------------------------------------------------ training
def train(cond, beta, data, val, shapes, metric_basis, seed=0, verbose=False):
    u, y, starts = data
    U, Y, L = make_batch(u, y, starts)
    th0 = torch.tensor(S.THETA_TRUE * DETUNE, dtype=F64)
    p_th = torch.zeros(4, dtype=F64, requires_grad=True)
    theta_of = lambda: th0 * torch.exp(p_th)
    eta0, _ = G.init_eta(seed)
    eta = eta0.clone().requires_grad_(True)
    xi = torch.zeros(NX * 2 * N_LAG + NX, dtype=F64)
    X0 = torch.tensor(S.baseline_states(u)[starts], dtype=F64)   # BUG FIX 2026-09-07
    xi[:NX * 2 * N_LAG] = torch.linalg.lstsq(L, X0).solution.T.reshape(-1)
    xi = xi.requires_grad_(True)

    params = [p_th, xi] if cond == 0 else [p_th, eta, xi]
    opt = torch.optim.Adam(params, lr=LR)

    Qp = Zp = QS = MD = None
    if cond in (2, 5):
        Qp, Zp, _ = pointwise_basis(th0.clone(), u, y)
    if cond in (3, 5):
        QS, MD, _ = extension_basis(th0.clone(), eta.detach(), xi.detach(), U, L, shapes)

    for it in range(N_ITER):
        opt.zero_grad()
        mode = 'annoff' if cond == 0 else 'full'
        theta = theta_of()
        yh = rollout(theta, eta, xi, U, L, shapes, mode=mode)
        loss = ((yh - Y) ** 2).mean()
        if cond == 2:
            w = ann_b(Zp, G.unpack(eta, shapes))[:, :NX].reshape(-1)
            loss = loss + beta * (Qp.T @ w).pow(2).sum()
        elif cond == 3:
            # PENALISE THE TOTAL contribution, both paths at once.
            # `mode='annoff'` zeroes the whole ANN output, so `d` carries the direct
            # correction AND the additional-state path. An earlier version used
            # `mode='frozen'` (x_aug held at zero), which isolates the state path only: that
            # is the right object for a DIAGNOSTIC, since it is the path Gyorok misses, but
            # the wrong one for the penalty. Measured 2026-09-06 with 'frozen': r_state fell
            # 0.278 -> 0.0175 while r_total stayed at 0.21 in every condition, i.e. the
            # direct path was left completely free to substitute for the parameters.
            yz = rollout(theta, eta, xi, U, L, shapes, mode='annoff')
            dtot = (yh - yz).reshape(-1)
            loss = loss + beta * (QS.T @ (MD @ dtot)).pow(2).sum()
        elif cond == 4:
            yz = rollout(theta, eta, xi, U, L, shapes, mode='annoff')
            loss = loss + beta * (yh - yz).pow(2).sum()      # size only, no projection
        elif cond == 5:
            yz = rollout(theta, eta, xi, U, L, shapes, mode='annoff')
            dtot = (yh - yz).reshape(-1)
            w = ann_b(Zp, G.unpack(eta, shapes))[:, :NX].reshape(-1)
            loss = (loss + beta * (QS.T @ (MD @ dtot)).pow(2).sum()
                         + beta * (Qp.T @ w).pow(2).sum())
        loss.backward()
        opt.step()
        if verbose and (it + 1) % 500 == 0:
            print(f'      it {it+1}: loss {loss.item():.3e}', flush=True)

    # ---------------- metrics ----------------
    with torch.no_grad():
        th = theta_of().detach()
        kap = np.array([th[0].item(), th[1].item(), (th[2] + th[3]).item()])
        kap_t = np.array([S.M1, S.C1, S.KA_TRUE + S.KB_TRUE])
        kap_err = float(np.linalg.norm((kap - kap_t) / kap_t))

        uv, yv, sv = val
        Uv, Yv, Lv = make_batch(uv, yv, sv)
        yhv = rollout(th, eta.detach(), xi.detach(), Uv, Lv, shapes,
                      mode=('annoff' if cond == 0 else 'full'))
        rmse = float(((yhv - Yv) ** 2).mean().sqrt())

        if cond == 0:
            return dict(kappa_err=kap_err, val_rmse=rmse, ell_total=None, apar_total=0.0,
                        aperp_total=0.0, ell_state=None, apar_state=0.0, aperp_state=0.0,
                        norm_d=0.0, norm_dstate=0.0, kappa=kap.tolist())
        yo = rollout(th, eta.detach(), xi.detach(), Uv, Lv, shapes, mode='annoff')
        yz = rollout(th, eta.detach(), xi.detach(), Uv, Lv, shapes, mode='frozen')
        dvec = (yhv - yo).reshape(-1).numpy()
        dst = (yhv - yz).reshape(-1).numpy()
        QSv, MDv = metric_basis                     # computed ONCE, shared by every run

        def split(v, tol=1e-12):
            """Separate ORTHOGONALITY from SHRINKAGE, and from movement into encoder space.

            The previous ratio was ||Q^T M_D d|| / ||d||: projected numerator, UNprojected
            denominator. It therefore factorises as (overlap) x (fraction surviving encoder
            projection), so a fall in it did not mean improved orthogonality. Reported now:
              ell   overlap INSIDE the encoder-profiled space; unchanged by uniform shrinking
              a_par amplitude inside  span(Q_S)
              a_perp amplitude outside span(Q_S)
            Uniform shrinking lowers both amplitudes and leaves ell flat. Selective removal
            of the in-span component lowers ell. ell is undefined for a negligible vector.
            """
            vb = MDv @ v
            nb = float(np.linalg.norm(vb))
            a_par = float(np.linalg.norm(QSv.T @ vb))
            a_perp = float(np.linalg.norm(vb - QSv @ (QSv.T @ vb)))
            ell = a_par / nb if nb > tol * max(1.0, float(np.linalg.norm(v))) else float('nan')
            return ell, a_par, a_perp, nb

        l_t, ap_t, ae_t, nb_t = split(dvec)
        l_s, ap_s, ae_s, nb_s = split(dst)
        return dict(kappa_err=kap_err, val_rmse=rmse,
                    ell_total=l_t, apar_total=ap_t, aperp_total=ae_t,
                    ell_state=l_s, apar_state=ap_s, aperp_state=ae_s,
                    norm_d=float(np.linalg.norm(dvec)),
                    norm_dstate=float(np.linalg.norm(dst)), kappa=kap.tolist())


def main():
    torch.set_default_dtype(F64)
    t0 = time.time()
    print('=' * 78)
    print('TESTBED TRAINING COMPARISON.  Category: development tool, not gantry evidence.')
    print('=' * 78)

    d_tr = S.make_dataset(seed=0)
    d_va = S.make_dataset(seed=7)
    rng = np.random.default_rng(0)
    st_tr = rng.choice(np.arange(N_LAG, len(d_tr['u']) - T_WIN - 1), N_WIN, replace=False)
    st_va = rng.choice(np.arange(N_LAG, len(d_va['u']) - T_WIN - 1), N_WIN, replace=False)
    data = (d_tr['u'], d_tr['y'], st_tr)
    val = (d_va['u'], d_va['y'], st_va)
    _, shapes = G.init_eta(0)

    # Shared VALIDATION metric basis, built once at the reference point. Rebuilding it per run
    # was ~10 expensive Jacobians; it is also more faithful to the deployed method, which
    # anchors the basis at theta_bar rather than re-deriving it for each trained model.
    Uv0, _, Lv0 = make_batch(d_va['u'], d_va['y'], st_va)
    _eta0, _ = G.init_eta(0)
    _xi0 = torch.zeros(NX * 2 * N_LAG + NX, dtype=F64)
    _X0 = torch.tensor(S.baseline_states(d_va['u'])[st_va], dtype=F64)   # BUG FIX 2026-09-07
    _xi0[:NX * 2 * N_LAG] = torch.linalg.lstsq(Lv0, _X0).solution.T.reshape(-1)
    _QS, _MD, _ = extension_basis(torch.tensor(S.THETA_TRUE, dtype=F64), _eta0, _xi0,
                                  Uv0, Lv0, shapes)
    metric_basis = (_QS.numpy(), _MD.numpy())

    print(f'  theta init detuned by {[f"{v:+.0%}" for v in (DETUNE - 1)]}')
    print(f'  {N_ITER} Adam iters, lr {LR}, {N_WIN} windows x {T_WIN} steps, noiseless\n')

    rows, res = [], {'category': 'development tool; testbed only, not gantry evidence',
                     'detune': DETUNE.tolist(), 'n_iter': N_ITER, 'runs': []}
    names = {0: 'baseline only', 1: 'joint, no penalty', 2: 'joint + pointwise',
             3: 'joint + extension', 4: 'joint + size (ctrl)',
             5: 'joint + both'}
    for cond in (0, 1, 2, 3, 4, 5):
        betas = [0.0] if cond in (0, 1) else BETAS
        for b in betas:
            r = train(cond, b, data, val, shapes, metric_basis)
            r.update(cond=cond, name=names[cond], beta=b)
            res['runs'].append(r)
            rows.append(r)
            fm = lambda v: ' n/a  ' if v is None or v != v else f'{v:.4f}'
            print(f'  [{names[cond]:<19s} b={b:>6.0e}]  kap {r["kappa_err"]:.4f}  '
                  f'rmse {r["val_rmse"]:.2e} | tot: ell {fm(r["ell_total"])} '
                  f'a| {r["apar_total"]:.2e} a_ {r["aperp_total"]:.2e} | '
                  f'state: ell {fm(r["ell_state"])} a| {r["apar_state"]:.2e}', flush=True)

    base = [r for r in rows if r['cond'] == 0][0]
    print('\n' + '-' * 78)
    print('THE PREDICTION (gaps.md Sect. 0): orthogonality should leave the parameter estimate')
    print('no worse than the baseline-alone fit.')
    print(f'  baseline-alone kappa error      : {base["kappa_err"]:.4f}')
    for cond in (1, 2, 3, 4, 5):
        best = min((r for r in rows if r['cond'] == cond), key=lambda r: r['kappa_err'])
        rel = best['kappa_err'] / base['kappa_err']
        verdict = 'no worse' if rel <= 1.05 else ('WORSE' if rel > 1.2 else 'marginal')
        print(f'  {names[cond]:<20s} best   : {best["kappa_err"]:.4f}   '
              f'{rel:5.2f}x baseline   {verdict}')
    print('\n  Absolute ||d_state|| is reported so a penalty cannot look good merely by')
    print('  switching the additional states off. Check it before reading any ratio.')
    print(f'\n  wall clock {time.time()-t0:.1f} s')

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print(f'  wrote {OUT}')


if __name__ == '__main__':
    main()
