"""AB1: can the thesis MLP state map reach its own equation-error target, and how fast? (diagnostic)

Encoder states of a probe checkpoint (default f_inn.pt) on contiguous stretches of 4 training records give
pairs (z(k) = [x_hat(k), u(k)], x_hat_a(k+1)). The EE term of E3 asks F_a(z(k)) = x_hat_a(k+1).
  1. Fixed-feature ceiling: least squares of x_hat_a(k+1) on [h(z), 1], h = the MLP's current hidden layers;
     the added-row output layer at that optimum: |W| vs the current |W_a|, residual/var, A_aa poles.
  2. EE-only Adam fits on copies of the MLP (nothing else changes; rollout loss absent, so this is the EE
     sub-problem only): (i) added-row output layer at lr 1e-3 (E3 setting), (ii) at 1e-2,
     (iii) 1e-2 plus hidden layers at 1e-3; residual/var and A_aa poles every 250 updates.
Updates here are one-step evaluations on 4096 random pairs, not rollouts (cheap); the count is what transfers.

Usage: python ab_ee_fit.py [CKPT] [N_UPD]
"""
import copy
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402

N_SEG, N_REC, B = 3000, 4, 4096


def poles(fs, x, u):
    Ef, Ea = up.jac_eigs(fs, x, u)
    dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
    pa = up.band_pair(Ea)
    ok = np.any(~np.isnan(pa[:, 0]))
    return 'dom |z| %.3f, complex %.0f %%, band pair %.0f %% (f %.0f, zeta %.2f)' % (
        np.median(np.abs(dom)), 100 * np.mean(np.abs(Ea.imag).max(1) > 1e-9), 100 * np.mean(~np.isnan(pa[:, 0])),
        np.nanmedian(pa[:, 0]) if ok else np.nan, np.nanmedian(pa[:, 1]) if ok else np.nan)


def main(fname, n_upd):
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    st = torch.load(os.path.join(HERE, 'outputs', fname), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    order = np.argsort([-len(r[2]) for r in recs])[:N_REC]
    X0, U0, X1 = [], [], []
    for r in order:
        u, y, starts, row = recs[r]
        ks = np.arange(starts[0], starts[0] + N_SEG + 1)
        uh = np.stack([u[k - fs.nb:k + nb_r] for k in ks]); yh = np.stack([y[k - fs.na:k + na_r] for k in ks])
        with torch.no_grad():
            x = fs.encoder(torch.as_tensor(np.ascontiguousarray(uh), dtype=cfg.dtype_pt), torch.as_tensor(np.ascontiguousarray(yh), dtype=cfg.dtype_pt))
        X0.append(x[:-1]); X1.append(x[1:, 6:]); U0.append(torch.as_tensor(u[ks[:-1]], dtype=cfg.dtype_pt))
    X0, U0, X1 = torch.cat(X0), torch.cat(U0), torch.cat(X1)
    var = float(X1.var(0).sum())
    sub = np.arange(0, len(X0), 500)
    xs, us = X0[sub], U0[sub]
    net0 = ann.net
    print('\n######## AB1 EE fit [%s]: %d pairs, var(x_a+) %.3f, current |W_a| %.3f' % (
        fname, len(X0), var, net0.out_a.weight.norm().item()), flush=True)

    def fa(net):
        ann.net = net
        try:
            return fs.hfn(X0, U0)[1].reshape(len(X0), -1)[:, 6:8]
        finally:
            ann.net = net0

    with torch.no_grad():
        r = float((X1 - fa(net0)).pow(2).sum(1).mean()) / var
    print('  current model: EE residual/var %.3f | A_aa %s' % (r, poles(fs, xs, us)), flush=True)
    # 1. fixed-feature LS ceiling on the added-row output layer
    with torch.no_grad():
        z = torch.cat([X0, U0], 1)
        H = net0.hidden(z)
        sk = net0.skip(z) if net0.skip is not None else 0
        Hc = torch.cat([H, torch.ones(len(H), 1, dtype=H.dtype)], 1)
        W = torch.linalg.lstsq(Hc, X1 - sk).solution
        net1 = copy.deepcopy(net0)
        net1.out_a.weight.copy_(W[:-1].T); net1.out_a.bias.copy_(W[-1])
        r = float((X1 - fa(net1)).pow(2).sum(1).mean()) / var
        print('  hidden features: rms per unit %s' % np.round(H.std(0).numpy(), 2), flush=True)
    ann.net = net1
    print('  1. fixed-feature LS ceiling: |W| %.2f (now %.3f), EE residual/var %.4f | A_aa %s' % (
        net1.out_a.weight.norm().item(), net0.out_a.weight.norm().item(), r, poles(fs, xs, us)), flush=True)
    ann.net = net0
    # 2. EE-only Adam fits
    for name, lr_a, lr_h in (('out_a 1e-3', 1e-3, 0.0), ('out_a 1e-2', 1e-2, 0.0), ('out_a 1e-2 + hidden 1e-3', 1e-2, 1e-3)):
        net = copy.deepcopy(net0)
        groups = [{'params': list(net.out_a.parameters()), 'lr': lr_a}]
        if lr_h:
            groups.append({'params': list(net.hidden.parameters()), 'lr': lr_h})
        opt = torch.optim.Adam(groups, eps=cfg.adam_eps)
        rng = np.random.default_rng(0)
        print('  2. %s' % name, flush=True)
        for it in range(1, n_upd + 1):
            ix = torch.as_tensor(rng.integers(len(X0), size=B))
            ann.net = net
            try:
                l = (X1[ix] - fs.hfn(X0[ix], U0[ix])[1].reshape(B, -1)[:, 6:8]).pow(2).sum(1).mean() / var
                opt.zero_grad(); l.backward(); opt.step()
            finally:
                ann.net = net0
            if it in (50, 100, 250) or it % 500 == 0:
                with torch.no_grad():
                    r = float((X1 - fa(net)).pow(2).sum(1).mean()) / var
                ann.net = net
                print('     upd %5d: EE residual/var %.4f, |W_a| %.2f | A_aa %s' % (
                    it, r, net.out_a.weight.norm().item(), poles(fs, xs, us)), flush=True)
                ann.net = net0


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'f_inn.pt', int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
