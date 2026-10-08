"""Valley or barrier EARLY in training (unchanged thesis model), diagnostic only. Same protocol as y_valley.py
(Z1), but from a u_probe checkpoint of the zero-init model (U_INIT=zero) instead of run 42's end point.

A temporary measuring probe adds Delta x_a to the added rows, Delta(s) = s (A_T - A0), Delta frozen;
A0 = the added self-map averaged over the eval points, A_T = 0.95 R(212 Hz) (diagnostic destination).
Per s: re-fit N updates (added-row output layer 1e-3, everything else 1e-5), same start and batches.

Usage: python y2_valley_early.py TAG N_UPDATES BATCH
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402
from y_valley import S_VALUES                                                  # noqa: E402


class WithProbe(torch.nn.Module):
    """Wraps the probe's RowSplitNet; adds the frozen measuring probe Delta x_a on the added rows."""

    def __init__(self, net, n_phys=6):
        super().__init__()
        self.inner, self.n_phys = net, n_phys
        self.register_buffer('delta', torch.zeros(2, 2, dtype=net.out_a.weight.dtype))

    def forward(self, z):
        out = self.inner(z)
        xa = z[:, self.n_phys:self.n_phys + 2]
        return torch.cat([out[:, :self.n_phys], out[:, self.n_phys:] + xa @ self.delta.T], 1)


def main(tag, n_upd, B):
    cfg, data, norm, fs, ann, others = up.setup()
    st = torch.load(os.path.join(HERE, 'outputs', 'u_%s.pt' % tag), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    n_done = st['n_upd']
    inner = ann.net
    ann.net = WithProbe(inner)
    net = ann.net
    st = {'hfn': {k: v.clone() for k, v in fs.hfn.state_dict().items()},
          'enc': {k: v.clone() for k, v in fs.encoder.state_dict().items()}}
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 216, np.random.default_rng(321), cfg.dtype_pt)
    with torch.no_grad():
        l_ref = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
        x0 = fs.encoder(ev[0], ev[1])[:24]
    u0 = ev[2][:24, 0]
    f = lambda xx: fs.hfn(xx, u0)[1].reshape(x0.shape[0], -1)[:, 6:8].sum(0)          # noqa: E731
    A0 = torch.autograd.functional.jacobian(f, x0.clone()).permute(1, 0, 2)[:, :, 6:8].detach().mean(0)
    th = 2 * np.pi * 212 / up.FS
    A_T = 0.95 * torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=A0.dtype)
    print('\n######## early valley scan [%s, %d updates]: loss (216 windows) %.4e; A0 eig %s; %d updates per s, batch %d'
          % (tag, n_done, l_ref, np.round(np.linalg.eigvals(A0.numpy()), 4), n_upd, B), flush=True)
    for s in S_VALUES:
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        with torch.no_grad():
            net.delta.copy_(s * (A_T - A0))
            l_start = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
        opt = torch.optim.Adam([{'params': others, 'lr': 1e-5},
                                {'params': list(inner.hidden.parameters()) + list(inner.out_p.parameters()), 'lr': 1e-5},
                                {'params': list(inner.out_a.parameters()), 'lr': 1e-3}], lr=1e-5, eps=cfg.adam_eps)
        rng = np.random.default_rng(7); ls = []; t0 = time.time()
        for it in range(1, n_upd + 1):
            uh, yh, uf, yf, cix = n1.batch(recs, fs, na_r, nb_r, B, rng, cfg.dtype_pt)
            opt.zero_grad()
            loss = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=n1.NF)
            loss.backward(); opt.step()
            if it % 50 == 0 or it == n_upd:
                with torch.no_grad():
                    ls.append(float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF)))
        with torch.no_grad():
            x0 = fs.encoder(ev[0], ev[1])[:24]
        Ef, Ea = up.jac_eigs(fs, x0, u0)
        pa = up.band_pair(Ea)
        dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
        ok = np.any(~np.isnan(pa[:, 0]))
        print('  s %.2f: start %.4e -> %s | /ref %.3f | added eig: |z| med %.3f, dom %.0f Hz, band pair %.0f %% (f %.0f, zeta %.2f) | %.0f s' % (
            s, l_start, ' -> '.join('%.4e' % v for v in ls), ls[-1] / l_ref, np.median(np.abs(dom)),
            np.median(np.abs(np.angle(dom))) * up.FS / (2 * np.pi), 100 * np.mean(~np.isnan(pa[:, 0])),
            np.nanmedian(pa[:, 0]) if ok else np.nan, np.nanmedian(pa[:, 1]) if ok else np.nan, time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
