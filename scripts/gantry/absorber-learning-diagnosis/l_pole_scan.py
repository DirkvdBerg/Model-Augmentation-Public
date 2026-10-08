"""Loss landscape of the added pole (mechanism test after K1).

Starting from the K1 member (physics at true combinations + the 2x16 tanh net realising the
least-squares linear absorber map), the added self-map A = L[6:8, 6:8] = V diag(l, conj l) V^-1 is
replaced by V diag(r e^{i th}, conj) V^-1, everything else (input map B_a, output map C_a, static
rows) kept. The exact training loss (encoder physical rows, x_a = 0 at the window start, nf 400) is
evaluated on a grid of pole radius r and pole frequency f = th / (2 pi Ts).
Reading: flat in f at small r and a narrow well around 212 Hz near the unit circle = the gradient
has no frequency information far from the answer (Zucchet and Orvieto 2024, Hayes et al. 2023);
loss rising with r away from 212 Hz = a mis-tuned resonator is penalised, damping preferred.

Usage: python l_pole_scan.py [job] [n_windows_per_record]   (default 86894, 12)
"""
import os
import sys
import copy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS                                          # noqa: E402  (sets sys.path)
from h_learned_poles import build                                              # noqa: E402
from i_supervised_fit import ann_of, truth_step, physics_step                  # noqa: E402
from k_linear_member import set_linear                                         # noqa: E402

NF = 400
SUB = 6
TS = 1 / 4000.0
R_GRID = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.97, 0.9844, 0.99]
F_GRID = [150, 170, 190, 200, 206, 212, 218, 225, 240, 260, 290]


def main(job, npr):
    from scipy.io import loadmat
    from gantry_dynamic.data import load_mat_aug, traj_dir
    from gantry_dynamic.closed_loop_report import _true_free
    from common import oracle
    cfg, data, norm, fs = build(job)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)      # noqa: E731
    xm, xs = np.asarray(norm.x_mean).ravel(), np.asarray(norm.std_x).ravel()
    u0, us = np.asarray(fs.norm.u0).ravel(), np.asarray(fs.norm.ustd).ravel()
    up = int(cfg.hp['up_sample'])
    fm = copy.deepcopy(fs)
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
    for (u, y, xl, xa, tpr) in recs:                      # identical to k_linear_member.py
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
    L0, b = coef[:-1].T, coef[-1]
    lam, V = np.linalg.eig(L0[6:8, 6:8])
    print('\n######## pole scan, job %s: member pole |z| %.4f, %.1f Hz' % (job, abs(lam[0]), abs(np.angle(lam[0])) / (2 * np.pi * TS)), flush=True)

    # fixed windows: encoder physical rows, x_a = 0
    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    ctrl_rows = getattr(fs.simulator, 'train_ctrl_rows', None)
    W = []
    for r, sd in enumerate(data.train_list):
        sdn = fs.norm.transform(sd)
        uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF, na_right=na_r, nb_right=nb_r, stride=cfg.stride)[:4]
        pick = np.linspace(0, len(uf) - 1, npr).round().astype(int)
        uh, yh, uf, yf = (T(a[pick]) for a in (uh, yh, uf, yf))
        with torch.no_grad():
            xe = fs.encoder(uh, yh); xe[:, 6:] = 0
        row = 0 if ctrl_rows is None else int(np.asarray(ctrl_rows)[r])
        W.append((xe, uf, yf, torch.full((npr,), row, dtype=torch.long)))

    def loss_at(Lm):
        set_linear(ann_of(fm).net, Lm, b)
        tot, n = 0.0, 0
        with torch.no_grad():
            for xe, uf, yf, cix in W:
                yp, _ = fm.simulator(fm, xe, uf, yf, ctrl_ix=cix)
                e = (yp - yf).numpy()
                tot += np.sum(e ** 2); n += e.size
        return tot / n

    base = loss_at(L0)
    print('  member loss on %d windows: %.4e' % (npr * len(W), base), flush=True)
    grid = np.full((len(R_GRID), len(F_GRID)), np.nan)
    for i, rr in enumerate(R_GRID):
        for j, fq in enumerate(F_GRID):
            if rr == 0.0 and j > 0:
                grid[i, j] = grid[i, 0]; continue
            z = rr * np.exp(1j * 2 * np.pi * fq * TS)
            Aaa = np.real(V @ np.diag([z, np.conj(z)] if np.angle(lam[0]) > 0 else [np.conj(z), z]) @ np.linalg.inv(V))
            Lm = L0.copy(); Lm[6:8, 6:8] = Aaa
            grid[i, j] = loss_at(Lm)
        print('  r %.4f: %s' % (rr, ' '.join('%.3e' % v for v in grid[i])), flush=True)
    print('  columns f [Hz]: %s' % ' '.join('%9d' % f for f in F_GRID))
    print('  loss / member loss:')
    for i, rr in enumerate(R_GRID):
        print('  r %.4f: %s' % (rr, ' '.join('%9.2f' % v for v in grid[i] / base)))
    np.savez(os.path.join(HERE, 'outputs', 'l_scan_%s.npz' % job), grid=grid, r=R_GRID, f=F_GRID, base=base)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '86894', int(sys.argv[2]) if len(sys.argv) > 2 else 12)
