"""Representation test: CAN the model class (physics block + 2x16 tanh Static_ANN_Block writing 6
physical rows and n_a added rows) realise the truth's absorber, and does that solution score better
on the exact training loss than the trained model?

1. One-step targets from the truth (FP + absorber + tanh friction, RK4 with the model's up_sample)
   at the true states of every training record (every SUB-th sample):
     physical rows: x_phys+ (truth) minus the model's physics step (trained combinations, ANN off)
     added rows:    the truth's absorber state [delta_a, vdelta_a] at k+1, in a fixed gauge (/ std)
   all in the model's normalised coordinates.
2. Supervised fit of a FRESH ANN of the same architecture (zero-init output layer), Adam then
   L-BFGS, rows weighted (HEURISTIC label below). A warm start from the trained ANN stalled: the
   trained physical rows live in the encoder's gauge, not in the true state.
3. The fitted model: realised poles (as h_learned_poles.py) and the exact windowed training loss
   (g_window_loss.py windows, nf 400, closed-loop residual rollout) with
     fit_T1   true physical state + true absorber state at each window start
     fit_T2   true physical state, absorber state zero
     fit_enc  encoder physical rows, absorber state zero (what training could use)
   against the trained model (encoder x0, and encoder with x_a = 0).
Evaluation and a supervised fit only; no training-loss gradient step.

Usage: python i_supervised_fit.py [job] [SUB]   (default 86894, 6)
"""
import os
import sys
import copy
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS                                          # noqa: E402  (sets sys.path)
from b_error_budget import _deriv_b, _stack                                    # noqa: E402
from h_learned_poles import build, step_jac, zinfo, band_pairs                 # noqa: E402

N_PER_REC = 48          # HEURISTIC: same windows as g_window_loss.py
NF = 400


def ann_of(fs):
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    return next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))


def truth_step(x8, u_log, tp, ts, up):
    p = {k: np.full(len(x8), v) for k, v in tp.items()}
    x = x8.copy(); h = ts / up
    for _ in range(up):
        k1 = _deriv_b(x, u_log, p); k2 = _deriv_b(x + 0.5 * h * k1, u_log, p)
        k3 = _deriv_b(x + 0.5 * h * k2, u_log, p); k4 = _deriv_b(x + h * k3, u_log, p)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def physics_step(fs, xn, un):
    ann = ann_of(fs); g = ann.out_gate.clone()
    ann.out_gate = torch.zeros_like(g)
    try:
        with torch.no_grad():
            return fs.hfn(xn, un)[1].reshape(len(xn), -1)
    finally:
        ann.out_gate = g


def main(job, SUB):
    from scipy.io import loadmat
    from gantry_dynamic.data import load_mat_aug, traj_dir
    from common import oracle
    run, arm, na, seed = JOBS[job]
    assert na == 2 and arm == 'u'
    cfg, data, norm, fs = build(job)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)      # noqa: E731
    xm, xs = np.asarray(norm.x_mean).ravel(), np.asarray(norm.std_x).ravel()
    u0, us = np.asarray(fs.norm.u0).ravel(), np.asarray(fs.norm.ustd).ravel()
    up = int(cfg.hp['up_sample'])
    Lb = None

    # ---- 1. one-step targets on the true states
    recs = []
    for f in data.train_files:
        u, y, xl, xa = load_mat_aug(f, cfg)
        meta = loadmat(os.path.join(traj_dir(cfg), f), variable_names=['meta'], squeeze_me=True,
                       struct_as_record=False)['meta']
        tp = oracle.truth_from_meta(meta.truth)
        recs.append((u, y, xl, xa, tp))
    tp = recs[0][4]; Lb = tp['Lb']
    Pm = np.array([[1.0, 1.0, 0.0], [Lb / 2, -Lb / 2, 0.0], [0.0, 0.0, 1.0]])
    xa_all = np.concatenate([r[3] for r in recs])
    sa = xa_all.std(0)                                            # gauge of the added states
    Z, Tg = [], []
    for (u, y, xl, xa, tpr) in recs:
        ix = np.arange(0, len(u) - 1, SUB)
        x8 = np.concatenate([xl[ix, :3], xa[ix, :1], xl[ix, 3:], xa[ix, 1:]], 1)
        x8n = truth_step(x8, (Pm @ u[ix].T).T, tpr, cfg.ts_new, up)
        xn = np.concatenate([(xl[ix] - xm) / xs, xa[ix] / sa], 1)
        un = (u[ix] - u0) / us
        fp = physics_step(fs, T(xn), T(un)).numpy()
        nxt = np.concatenate([(np.concatenate([x8n[:, :3], x8n[:, 4:7]], 1) - xm) / xs,
                              np.stack([x8n[:, 3], x8n[:, 7]], 1) / sa], 1)
        tg = nxt.copy(); tg[:, :6] -= fp[:, :6]
        Z.append(np.concatenate([xn, un], 1)); Tg.append(tg)
    Z, Tg = T(np.concatenate(Z)), T(np.concatenate(Tg))
    # coordinate check: truth without absorber and friction vs the model physics at the TRUE
    # combinations, same points: must agree far below the target std if units/frames are right
    from gantry_dynamic.closed_loop_report import _without_learned, _true_free
    from b_error_budget import variant
    phy = next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params'))
    fbt = _without_learned(fs, _true_free(phy))
    (u, y, xl, xa, tpr) = recs[0]
    ix = np.arange(0, len(u) - 1, SUB)
    x8 = np.concatenate([xl[ix, :3], np.zeros((len(ix), 1)), xl[ix, 3:], np.zeros((len(ix), 1))], 1)
    xb = truth_step(x8, (Pm @ u[ix].T).T, variant(tpr, 'bt'), cfg.ts_new, up)
    xn = np.concatenate([(xl[ix] - xm) / xs, np.zeros((len(ix), 2))], 1)
    fb = physics_step(fbt, T(xn), T((u[ix] - u0) / us)).numpy()[:, :6]
    nb = (np.concatenate([xb[:, :3], xb[:, 4:7]], 1) - xm) / xs
    print('  coordinate check (bt truth vs physics at true combos), rms diff / target std per row: %s' % ' '.join(
        '%.2e' % v for v in (np.sqrt(np.mean((nb - fb) ** 2, 0)) / Tg[:, :6].std(0).numpy()).tolist()), flush=True)
    # HEURISTIC: row weights = 1/std, position rows x0.1 (implied by the velocity rows to O(Ts)),
    # added rows x30 (their self-map sets the pole; 1e-3 relative accuracy needed at |z| 0.986)
    w = 1.0 / Tg.std(0) * T(np.array([0.1, 0.1, 0.1, 1, 1, 1, 30, 30]))
    ann = ann_of(fs)
    with torch.no_grad():
        r_tr = ((ann.net(Z) - Tg) / Tg.std(0))
    print('\n######## run %d (job %s): %d one-step points (every %d-th sample, 18 records), gauge sa %s'
          % (run, job, len(Z), SUB, sa))
    print('  target std per row: %s' % ' '.join('%.3e' % v for v in Tg.std(0).tolist()))
    print('  trained ANN, RELATIVE one-step residual rms per row: %s' % ' '.join('%.3e' % v for v in r_tr.pow(2).mean(0).sqrt().tolist()))

    # ---- 2. supervised fit (warm start)
    from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
    torch.manual_seed(0)
    net = zero_init_feed_forward_nn(n_in=Z.shape[1], n_out=8, n_nodes_per_layer=cfg.n_nodes_per_layer,
                                    n_hidden_layers=cfg.n_hidden_layers).double()
    assert [p.shape for p in net.parameters()] == [p.shape for p in ann.net.parameters()]
    adam = torch.optim.Adam(net.parameters(), lr=3e-3)
    t0 = time.time()
    for it in range(int(os.environ.get('I_ADAM', 3000))):
        adam.zero_grad(); l = (((net(Z) - Tg) * w) ** 2).mean(); l.backward(); adam.step()
        if it % 250 == 0:
            with torch.no_grad():
                rr = (net(Z) - Tg).pow(2).mean(0).sqrt() / Tg.std(0)
            print('  adam %5d  loss %.4e  rel rows %s  (%.0f s)' % (it, float(l), ' '.join('%.1e' % v for v in rr.tolist()), time.time() - t0), flush=True)
    opt = torch.optim.LBFGS(net.parameters(), lr=1, max_iter=20, history_size=50, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        l = (((net(Z) - Tg) * w) ** 2).mean()
        l.backward()
        return l
    t0 = time.time()
    for it in range(int(os.environ.get('I_OUTER', 100))):
        l = opt.step(closure)
        if it % 25 == 0:
            with torch.no_grad():
                rr = (net(Z) - Tg).pow(2).mean(0).sqrt() / Tg.std(0)
            print('  lbfgs %4d  loss %.4e  rows %s  (%.0f s)' % (it * 20, float(l), ' '.join('%.1e' % v for v in rr.tolist()), time.time() - t0), flush=True)
    with torch.no_grad():
        rr = (net(Z) - Tg).pow(2).mean(0).sqrt() / Tg.std(0)
    print('  fitted, RELATIVE (rms / target std) one-step residual rms per row: %s' % ' '.join('%.3e' % v for v in rr.tolist()))
    fsf = copy.deepcopy(fs)
    ann_of(fsf).net.load_state_dict(net.state_dict())
    torch.save(net.state_dict(), os.path.join(HERE, 'outputs', 'i_fitted_net_%s.pt' % job))

    # ---- 3a. realised poles of the fitted model at the true states (subsample)
    sel = torch.arange(0, len(Z), max(1, len(Z) // 600))
    J = step_jac(fsf, Z[sel, :8].clone(), Z[sel, 8:].clone())
    ev = [np.linalg.eigvals(j) for j in J]
    bp = [band_pairs(e) for e in ev]
    flat = np.concatenate([np.array(b) for b in bp if len(b)]) if any(len(b) for b in bp) else np.array([])
    print('  fitted model: points with a pair in 95 to 380 Hz: %d of %d' % (sum(len(b) > 0 for b in bp), len(bp)))
    if len(flat):
        fq, zt = zinfo(flat)
        for q in (10, 50, 90):
            print('     pct %2d: |z| %.4f  f %.1f Hz  zeta %.3f' % (q, np.percentile(np.abs(flat), q), np.percentile(fq, q), np.percentile(zt, q)))
    Jaa = J[:, 6:, 6:]
    eva = np.array([np.linalg.eigvals(j) for j in Jaa])
    print('  fitted A_aa eig median: |z| %s angle %s' % (np.median(np.abs(eva), 0), np.median(np.abs(np.angle(eva)), 0)))

    # ---- 3b. exact windowed training loss
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    ctrl_rows = getattr(fs.simulator, 'train_ctrl_rows', None)
    res = {k: [] for k in ('trained_enc', 'trained_enc_xa0', 'fit_T1', 'fit_T2', 'fit_enc')}
    for r, (f, sd) in enumerate(zip(data.train_files, data.train_list)):
        sdn = fs.norm.transform(sd)
        uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF, na_right=na_r, nb_right=nb_r, stride=cfg.stride)[:4]
        pick = np.linspace(0, len(uf) - 1, N_PER_REC).round().astype(int)
        uh, yh, uf, yf = (T(a[pick]) for a in (uh, yh, uf, yf))
        k = max(fs.na, fs.nb) + pick * cfg.stride
        xl, xa = recs[r][2], recs[r][3]
        xtrue = T(np.concatenate([(xl[k] - xm) / xs, xa[k] / sa], 1))
        row = 0 if ctrl_rows is None else int(np.asarray(ctrl_rows)[r])
        cix = torch.full((len(pick),), row, dtype=torch.long)
        with torch.no_grad():
            xenc = fs.encoder(uh, yh)
            xenc0 = xenc.clone(); xenc0[:, 6:] = 0
            xt0 = xtrue.clone(); xt0[:, 6:] = 0
            for name, m, x0 in (('trained_enc', fs, xenc), ('trained_enc_xa0', fs, xenc0),
                                ('fit_T1', fsf, xtrue), ('fit_T2', fsf, xt0), ('fit_enc', fsf, xenc0)):
                yp, _ = m.simulator(m, x0, uf, yf, ctrl_ix=cix)
                res[name].append((yp - yf).numpy())
        print('  %s done' % f, flush=True)
    ystd = np.asarray(norm.ystd, dtype=float).ravel()
    print('\n  windowed training loss (mse, normalised), %d windows, nf %d; reference: model 5.09e-9, oracle T1 5.1e-11, T2 1.35e-9, noabs 1.25e-8' % (
        N_PER_REC * len(data.train_files), NF))
    out = {}
    for kk, v in res.items():
        e = np.concatenate(v)
        fin = np.isfinite(e).all(axis=(1, 2))
        out[kk] = e
        print('  %-16s mse %.4e | Y rms %.3e m (steps 0-40 %.3e, 40-400 %.3e) | finite windows %d/%d' % (
            kk, np.mean(e[fin] ** 2), np.sqrt(np.mean(e[fin][..., 2] ** 2)) * ystd[2],
            np.sqrt(np.mean(e[fin][:, :40, 2] ** 2)) * ystd[2], np.sqrt(np.mean(e[fin][:, 40:, 2] ** 2)) * ystd[2], fin.sum(), len(fin)), flush=True)
    np.savez(os.path.join(HERE, 'outputs', 'i_fit_%s.npz' % job), sa=sa, w=w.numpy(),
             **{'mse_' + k: np.mean(v ** 2, axis=(1, 2)) for k, v in out.items()})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '86894', int(sys.argv[2]) if len(sys.argv) > 2 else 6)
