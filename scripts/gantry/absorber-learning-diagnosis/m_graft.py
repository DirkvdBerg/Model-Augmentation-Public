"""Does the trained static imitation BLOCK a resonator? (mechanism test after L1)

The trained model (run 42) gets two EXTRA states x_g carrying the K1 member's absorber:
  x+   = F_trained(x, u) + alpha * C_g x_g          (C_g = member's x_a columns on the physical rows)
  x_g+ = A_g x_g + B_g [x_phys, u] + b_g             (member's added rows)
F_trained is untouched (its own two fast added states included). A diagnostic model OUTSIDE the class
(10 states), used only to read the training loss against the graft strength alpha and its pole
radius r (at the member's 212 Hz angle). Same closed-loop residual rollout, windows, controller.
Reading: loss falling from alpha = 0 = a descent direction toward the resonator exists at the trained
point (not a basin; the cause is how the MLP reaches that direction). Loss rising at small alpha =
the static imitation occupies the absorber's place (the trained point is a basin against it).

Hosts: 'trained' (run 42 as trained) and 'phys_trained' (the same model with its learned block
removed: physics at the TRAINED combinations only, i.e. the static correction taken away) and
'phys_true' (learned block removed, TRUE combinations: neither static correction nor drift).
3rd argument: comma list of hosts. GRAFT_FULL=1: the graft also carries the member's static linear
terms on the physical rows, so that phys_true at alpha 1 IS the K1 member (construction check).

Usage: python m_graft.py [job] [n_windows_per_record]   (default 86894, 12)
"""
import os
import sys
import copy

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import JOBS                                          # noqa: E402,F401  (sets sys.path)
from h_learned_poles import build                                              # noqa: E402
from i_supervised_fit import truth_step, physics_step                          # noqa: E402

NF = 400
SUB = 6
TS = 1 / 4000.0
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
RADII = [0.6, 0.9, 0.9844]


class Graft(torch.nn.Module):
    """hfn-compatible step on [x (8), x_g (2)]; output from x only."""

    def __init__(self, hfn, Ag, Bg, bg, Cg, alpha, Pz=None, bp=None):
        super().__init__()
        self.h, self.alpha, self.Ag, self.Bg, self.bg, self.Cg = hfn, alpha, Ag, Bg, bg, Cg
        self.Pz, self.bp = Pz, bp          # GRAFT_FULL: the member's static linear terms on the physical rows

    def forward(self, x, u):
        x = x.reshape(x.shape[0], -1); u = u.reshape(u.shape[0], -1)
        xm, xg = x[:, :8], x[:, 8:]
        y, xn = self.h(xm, u)
        xn = xn.reshape(x.shape[0], -1)
        xgn = xg @ self.Ag.T + torch.cat([xm[:, :6], u], 1) @ self.Bg.T + self.bg
        corr = xg @ self.Cg.T
        if self.Pz is not None:
            corr = corr + torch.cat([xm[:, :6], u], 1) @ self.Pz.T + self.bp
        xn = xn + self.alpha * torch.cat([corr, torch.zeros_like(xn[:, 6:])], 1)
        return y, torch.cat([xn, xgn], 1)

    def output_only(self, x):
        return self.h.output_only(x.reshape(x.shape[0], -1)[:, :8])


def main(job, npr):
    from scipy.io import loadmat
    from gantry_dynamic.data import load_mat_aug, traj_dir
    from gantry_dynamic.closed_loop_report import _true_free
    from model_augmentation.fit_systems.closed_loop import closed_loop_rollout
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
    L, b = coef[:-1].T, coef[-1]
    Bg = T(np.concatenate([L[6:8, :6], L[6:8, 8:]], 1)); bg = T(b[6:8]); Cg = T(L[:6, 6:8])
    FULL = os.environ.get('GRAFT_FULL') == '1'
    Pz = T(np.concatenate([L[:6, :6], L[:6, 8:]], 1)) if FULL else None
    bp = T(b[:6]) if FULL else None
    print('  graft: %s' % ('FULL (member static rows + absorber)' if FULL else 'x_a readout only'))
    lam, V = np.linalg.eig(L[6:8, 6:8])
    print('\n######## graft, job %s: member pole |z| %.4f %.1f Hz' % (
        job, abs(lam[0]), abs(np.angle(lam[0])) / (2 * np.pi * TS)), flush=True)

    na_r, nb_r = getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0)
    ctrl_rows = getattr(fs.simulator, 'train_ctrl_rows', None)
    bank = fs.simulator.bank
    W = []
    for r, sd in enumerate(data.train_list):
        sdn = fs.norm.transform(sd)
        uh, yh, uf, yf = sdn.to_hist_future_data(na=fs.na, nb=fs.nb, nf=NF, na_right=na_r, nb_right=nb_r, stride=cfg.stride)[:4]
        pick = np.linspace(0, len(uf) - 1, npr).round().astype(int)
        uh, yh, uf, yf = (T(a[pick]) for a in (uh, yh, uf, yf))
        with torch.no_grad():
            xe = fs.encoder(uh, yh)
        row = 0 if ctrl_rows is None else int(np.asarray(ctrl_rows)[r])
        W.append((torch.cat([xe, torch.zeros(len(pick), 2, dtype=xe.dtype)], 1), uf, yf,
                  torch.full((npr,), row, dtype=torch.long)))

    from gantry_dynamic.closed_loop_report import _without_learned
    hosts = {'trained': fs.hfn, 'phys_trained': _without_learned(fs).hfn,
             'phys_true': _without_learned(fs, _true_free(next(b for b in fs.hfn.connected_blocks if hasattr(b, 'free_params')))).hfn}

    def loss(alpha, rad, host='trained'):
        z = rad * np.exp(1j * abs(np.angle(lam[0])))
        Ag = np.real(V @ np.diag([z, np.conj(z)] if np.angle(lam[0]) > 0 else [np.conj(z), z]) @ np.linalg.inv(V))
        g = Graft(hosts[host], T(Ag), Bg, bg, Cg, alpha, Pz, bp)
        tot, n = 0.0, 0
        with torch.no_grad():
            for x0, uf, yf, cix in W:
                yp, _, _ = closed_loop_rollout(g, g.output_only, uf, yf, x0, bank, cix)
                e = (yp - yf).numpy(); tot += np.sum(e ** 2); n += e.size
        return tot / n

    out = {}
    for host in (sys.argv[3].split(',') if len(sys.argv) > 3 else ('trained', 'phys_trained')):
        base = loss(0.0, 0.9844, host)
        print('\n  host %s alone (alpha 0) on %d windows: %.4e' % (host, npr * len(W), base), flush=True)
        grid = np.full((len(RADII), len(ALPHAS)), np.nan)
        for i, rad in enumerate(RADII):
            for j, al in enumerate(ALPHAS):
                grid[i, j] = base if al == 0 else loss(al, rad, host)
            print('  r %.4f: %s' % (rad, ' '.join('%.3e (%.3f)' % (v, v / base) for v in grid[i])), flush=True)
        print('  columns alpha: %s' % ALPHAS)
        out[host + '_grid'], out[host + '_base'] = grid, base
    np.savez(os.path.join(HERE, 'outputs', 'm_graft_%s%s_%s.npz' % (job, '_full' if FULL else '', '_'.join(k[:-5] for k in out if k.endswith('_grid')))), alphas=ALPHAS, radii=RADII, **out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '86894', int(sys.argv[2]) if len(sys.argv) > 2 else 12)
