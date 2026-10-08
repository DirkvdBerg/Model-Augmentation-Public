"""Valley or barrier? Profile scan from the trained thesis model (run 42, unchanged), diagnostic only.

A temporary MEASURING PROBE adds Delta x_a to the added rows (x_a+ = f_a(z) + Delta x_a), Delta frozen.
Path: Delta(s) = s (A_T - A0), A0 = the trained added self-map dF_a/dx_a averaged over the eval points,
A_T = 0.95 R(2 pi 212 Ts) (a diagnostic destination, as X1; never a method). For each s the rest of the
model (ANN, combinations, encoder) re-fits for N updates, same start and same batches for every s:
added-row output layer at lr 1e-3, everything else 1e-5 (the X1 protocol; the final MLP layer is split by
rows with the TRAINED weights copied, so the function is unchanged). Profile loss vs s:
  monotone falling from s = 0 -> VALLEY (an optimiser that follows the coupled pole/readout direction can
  walk it); rising at small s, falling later -> BARRIER (height = max over s of loss / loss(s=0)).
Also logged per s after the re-fit: realised added eigenvalues (median over points) and the share of points
with a complex pair in 95 to 380 Hz.

Usage: python y_valley.py N_UPDATES BATCH
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h_learned_poles as hp_                                                  # noqa: E402
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402

S_VALUES = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


class TrainedSplit(torch.nn.Module):
    """Trained zero-init MLP, final layer split by rows (weights copied), plus a frozen probe Delta."""

    def __init__(self, net, n_phys):
        super().__init__()
        layers = list(net.net)
        last = layers[-1]
        self.hidden = torch.nn.Sequential(*layers[:-1])
        self.out_p = torch.nn.Linear(last.in_features, n_phys).to(last.weight.dtype)
        self.out_a = torch.nn.Linear(last.in_features, last.out_features - n_phys).to(last.weight.dtype)
        with torch.no_grad():
            self.out_p.weight.copy_(last.weight[:n_phys]); self.out_p.bias.copy_(last.bias[:n_phys])
            self.out_a.weight.copy_(last.weight[n_phys:]); self.out_a.bias.copy_(last.bias[n_phys:])
        self.n_phys = n_phys
        self.register_buffer('delta', torch.zeros(2, 2, dtype=last.weight.dtype))

    def forward(self, z):
        h = self.hidden(z)
        xa = z[:, self.n_phys:self.n_phys + 2]
        return torch.cat([self.out_p(h), self.out_a(h) + xa @ self.delta.T], 1)


def main(n_upd, B):
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    cfg, data, norm, fs = hp_.build('86894')
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    old = {id(p) for p in ann.net.parameters()}
    others = [p for g in fs.optimizer.param_groups for p in g['params'] if id(p) not in old]
    ann.net = TrainedSplit(ann.net, 6)
    net = ann.net
    st = {'hfn': {k: v.clone() for k, v in fs.hfn.state_dict().items()},
          'enc': {k: v.clone() for k, v in fs.encoder.state_dict().items()}}
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 216, np.random.default_rng(321), cfg.dtype_pt)
    with torch.no_grad():
        l_ref = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
        x0 = fs.encoder(ev[0], ev[1])[:24]
    u0 = ev[2][:24, 0]
    Ef, Ea = up.jac_eigs(fs, x0, u0)
    f = lambda xx: fs.hfn(xx, u0)[1].reshape(x0.shape[0], -1)[:, 6:8].sum(0)          # noqa: E731
    J = torch.autograd.functional.jacobian(f, x0.clone()).permute(1, 0, 2)[:, :, 6:8].detach()
    A0 = J.mean(0)
    th = 2 * np.pi * 212 / up.FS
    A_T = 0.95 * torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=A0.dtype)
    print('\n######## valley scan, run 42: loss (216 windows) %.4e (checkpoint reproduced?); A0 mean\n%s\n'
          '  A0 eig %s; A_T eig %s; %d updates per s, batch %d' % (
              l_ref, A0.numpy(), np.round(np.linalg.eigvals(A0.numpy()), 4),
              np.round(np.linalg.eigvals(A_T.numpy()), 4), n_upd, B), flush=True)
    for s in S_VALUES:
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        with torch.no_grad():
            net.delta.copy_(s * (A_T - A0))
            l_start = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
        opt = torch.optim.Adam([{'params': others, 'lr': 1e-5},
                                {'params': list(net.hidden.parameters()) + list(net.out_p.parameters()), 'lr': 1e-5},
                                {'params': list(net.out_a.parameters()), 'lr': 1e-3}], lr=1e-5, eps=cfg.adam_eps)
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
        print('  s %.2f: start %.4e -> %s | /ref %.3f | added eig: |z| med %.3f, dom %.0f Hz, band pair %.0f %% (f %.0f, zeta %.2f) | %.0f s' % (
            s, l_start, ' -> '.join('%.4e' % v for v in ls), ls[-1] / l_ref, np.median(np.abs(dom)),
            np.median(np.abs(np.angle(dom))) * up.FS / (2 * np.pi), 100 * np.mean(~np.isnan(pa[:, 0])),
            np.nanmedian(pa[:, 0]) if np.any(~np.isnan(pa[:, 0])) else np.nan,
            np.nanmedian(pa[:, 1]) if np.any(~np.isnan(pa[:, 0])) else np.nan, time.time() - t0), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
