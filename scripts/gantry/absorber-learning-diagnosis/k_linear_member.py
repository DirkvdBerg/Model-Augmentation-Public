"""Representation test by construction (replaces I1's nonlinear fit): an explicit MEMBER of the thesis
model class that carries the absorber, built with no optimiser.

Member: the physics block at the TRUE combinations (a legal parameter value) plus the SAME 2x16 tanh
Static_ANN_Block, weights set so that it realises a linear map L of z = [x (8), u (3)]:
  W1 = eps [I_11; 0], W2 = eps [I_11 0; 0 0], W3 = L / eps^2   (tanh(e z) = e z (1 - (e z)^2 / 3 ...))
THEORY: small-signal linearisation of tanh; relative error about (eps |z|)^2 / 3 (checked numerically).
L = linear least squares of the truth's one-step correction on z over every SUB-th training sample:
  physical rows: truth (FP + absorber + tanh friction) step minus the physics step at true combinations
  added rows:    [delta_a, vdelta_a](k+1) / std   (the gauge of the added states)
Variants scored on the exact windowed training loss (g_window_loss windows):
  lin_T1 true state incl. absorber; lin_T2 true physical state, x_a = 0; lin_enc encoder physical rows
  (the trained encoder), x_a = 0; static_T2 = the same member with the added rows' output zeroed.
Plus the realised poles (as h_learned_poles.py).

Usage: python k_linear_member.py [job] [SUB]   (default 86894, 6)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS                                          # noqa: E402  (sets sys.path)
from h_learned_poles import build, step_jac, zinfo, band_pairs                 # noqa: E402
from i_supervised_fit import ann_of, truth_step, physics_step                  # noqa: E402

N_PER_REC = 48          # HEURISTIC: same windows as g_window_loss.py
NF = 400
EPS = 1e-4              # HEURISTIC: small-signal scale; |z| <= ~10 gives relative error <= 3e-6


def set_linear(net, L, b):
    """Realise z -> L z + b with a 2-hidden-layer tanh MLP (n_in <= width)."""
    lin = [l for l in net.net if isinstance(l, torch.nn.Linear)]
    assert len(lin) == 3
    n_in, width = lin[0].in_features, lin[0].out_features
    with torch.no_grad():
        for l in lin:
            l.weight.zero_(); l.bias.zero_()
        lin[0].weight[:n_in, :n_in] = EPS * torch.eye(n_in, dtype=lin[0].weight.dtype)
        lin[1].weight[:n_in, :n_in] = EPS * torch.eye(n_in, dtype=lin[1].weight.dtype)
        lin[2].weight[:, :n_in] = torch.as_tensor(L, dtype=lin[2].weight.dtype) / EPS ** 2
        lin[2].bias[:] = torch.as_tensor(b, dtype=lin[2].bias.dtype)


def main(job, SUB):
    import copy
    from scipy.io import loadmat
    from gantry_dynamic.data import load_mat_aug, traj_dir
    from gantry_dynamic.closed_loop_report import _true_free
    from common import oracle
    run, arm, na, seed = JOBS[job]
    assert na == 2 and arm == 'u'
    cfg, data, norm, fs = build(job)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)      # noqa: E731
    xm, xs = np.asarray(norm.x_mean).ravel(), np.asarray(norm.std_x).ravel()
    u0, us = np.asarray(fs.norm.u0).ravel(), np.asarray(fs.norm.ustd).ravel()
    up = int(cfg.hp['up_sample'])
    fm = copy.deepcopy(fs)                                                   # the member
    phy = next(b for b in fm.hfn.connected_blocks if hasattr(b, 'free_params'))
    with torch.no_grad():
        phy.free_params.copy_(_true_free(phy))

    recs = []
    for f in data.train_files:
        u, y, xl, xa = load_mat_aug(f, cfg)
        meta = loadmat(os.path.join(traj_dir(cfg), f), variable_names=['meta'], squeeze_me=True,
                       struct_as_record=False)['meta']
        recs.append((u, y, xl, xa, oracle.truth_from_meta(meta.truth)))
    Lb = recs[0][4]['Lb']
    Pm = np.array([[1.0, 1.0, 0.0], [Lb / 2, -Lb / 2, 0.0], [0.0, 0.0, 1.0]])
    sa = np.concatenate([r[3] for r in recs]).std(0)
    Z, Tg = [], []
    for (u, y, xl, xa, tpr) in recs:
        ix = np.arange(0, len(u) - 1, SUB)
        x8 = np.concatenate([xl[ix, :3], xa[ix, :1], xl[ix, 3:], xa[ix, 1:]], 1)
        x8n = truth_step(x8, (Pm @ u[ix].T).T, tpr, cfg.ts_new, up)
        xn = np.concatenate([(xl[ix] - xm) / xs, xa[ix] / sa], 1)
        un = (u[ix] - u0) / us
        fp = physics_step(fm, T(xn), T(un)).numpy()
        nxt = np.concatenate([(np.concatenate([x8n[:, :3], x8n[:, 4:7]], 1) - xm) / xs,
                              np.stack([x8n[:, 3], x8n[:, 7]], 1) / sa], 1)
        tg = nxt.copy(); tg[:, :6] -= fp[:, :6]
        Z.append(np.concatenate([xn, un], 1)); Tg.append(tg)
    Z, Tg = np.concatenate(Z), np.concatenate(Tg)
    A = np.concatenate([Z, np.ones((len(Z), 1))], 1)
    coef, *_ = np.linalg.lstsq(A, Tg, rcond=None)
    L, b = coef[:-1].T, coef[-1]
    res = Tg - A @ coef
    print('\n######## run %d (job %s): linear member, %d points, |z| max %.1f' % (run, job, len(Z), np.abs(Z).max()))
    print('  LS relative one-step residual per row: %s' % ' '.join('%.2e' % v for v in (res.std(0) / Tg.std(0)).tolist()))
    print('  LS added self-map L[6:8, 6:8] eig: %s' % ' '.join('|z| %.4f ang %.3f (%.1f Hz, zeta %.3f)' % (
        (abs(z), np.angle(z)) + zinfo(z)) for z in np.linalg.eigvals(L[6:8, 6:8]) if np.angle(z) >= 0))
    net = ann_of(fm).net
    set_linear(net, L, b)
    with torch.no_grad():
        got = net(T(Z)).numpy()
    print('  MLP realisation vs LS map: max rel err per row %s' % ' '.join('%.1e' % v for v in (
        np.abs(got - A @ coef).max(0) / Tg.std(0)).tolist()))
    sel = np.arange(0, len(Z), max(1, len(Z) // 600))
    J = step_jac(fm, T(Z[sel, :8]), T(Z[sel, 8:]))
    bp = [band_pairs(np.linalg.eigvals(j)) for j in J]
    flat = np.concatenate([np.array(q) for q in bp if len(q)]) if any(len(q) for q in bp) else np.array([])
    print('  member model: points with a pair in 95 to 380 Hz: %d of %d' % (sum(len(q) > 0 for q in bp), len(bp)))
    if len(flat):
        fq, zt = zinfo(flat)
        print('     median |z| %.4f  f %.1f Hz  zeta %.3f (p10-p90 zeta %.3f to %.3f)' % (
            np.median(np.abs(flat)), np.median(fq), np.median(zt), np.percentile(zt, 10), np.percentile(zt, 90)))
    fstat = copy.deepcopy(fm)
    with torch.no_grad():
        [l for l in ann_of(fstat).net.net if isinstance(l, torch.nn.Linear)][-1].weight[6:].zero_()
        [l for l in ann_of(fstat).net.net if isinstance(l, torch.nn.Linear)][-1].bias[6:].zero_()

    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    ctrl_rows = getattr(fs.simulator, 'train_ctrl_rows', None)
    out = {k: [] for k in ('lin_T1', 'lin_T2', 'lin_enc', 'static_T2')}
    for r, (f, sd) in enumerate(zip(data.train_files, data.train_list)):
        sdn = fs.norm.transform(sd)
        uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF, na_right=na_r, nb_right=nb_r, stride=cfg.stride)[:4]
        pick = np.linspace(0, len(uf) - 1, N_PER_REC).round().astype(int)
        uh, yh, uf, yf = (T(a[pick]) for a in (uh, yh, uf, yf))
        k = max(fs.na, fs.nb) + pick * cfg.stride
        xl, xa = recs[r][2], recs[r][3]
        xt = T(np.concatenate([(xl[k] - xm) / xs, xa[k] / sa], 1))
        xt0 = xt.clone(); xt0[:, 6:] = 0
        row = 0 if ctrl_rows is None else int(np.asarray(ctrl_rows)[r])
        cix = torch.full((len(pick),), row, dtype=torch.long)
        with torch.no_grad():
            xe = fs.encoder(uh, yh); xe[:, 6:] = 0
            for name, m, x0 in (('lin_T1', fm, xt), ('lin_T2', fm, xt0), ('lin_enc', fm, xe), ('static_T2', fstat, xt0)):
                yp, _ = m.simulator(m, x0, uf, yf, ctrl_ix=cix)
                out[name].append((yp - yf).numpy())
        print('  %s done' % f, flush=True)
    ystd = np.asarray(norm.ystd, dtype=float).ravel()
    print('\n  windowed training loss, 864 windows, nf 400; reference: trained 5.09e-9, oracle T1 5.1e-11, T2 1.35e-9, noabs 1.25e-8')
    for kk, v in out.items():
        e = np.concatenate(v); fin = np.isfinite(e).all(axis=(1, 2))
        print('  %-10s mse %.4e | Y rms %.3e m (steps 0-40 %.3e, 40-400 %.3e) | finite %d/%d' % (
            kk, np.mean(e[fin] ** 2), np.sqrt(np.mean(e[fin][..., 2] ** 2)) * ystd[2],
            np.sqrt(np.mean(e[fin][:, :40, 2] ** 2)) * ystd[2], np.sqrt(np.mean(e[fin][:, 40:, 2] ** 2)) * ystd[2],
            fin.sum(), len(fin)), flush=True)
    np.savez(os.path.join(HERE, 'outputs', 'k_member_%s.npz' % job), L=L, b=b, sa=sa,
             **{'mse_' + k: np.mean(np.concatenate(v) ** 2, axis=(1, 2)) for k, v in out.items()})


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '86894', int(sys.argv[2]) if len(sys.argv) > 2 else 6)
