"""Short local training probe with continuous pole tracking (handoff 2026-10-01, ladder of section 1).

The thesis model of run 42 (U arm, n_a 2, seed 1, detuned start, noisy data) with Jan's Static_ANN_Block,
routing and single 2x16 tanh MLP UNCHANGED as a function class and at initialisation. The only structural
edit is bookkeeping: the MLP's final Linear is split by rows into two zero-initialised Linear layers
(physical rows 0-5, added rows 6..) whose concatenation is the original layer, so the added-state writer
can get its own Adam parameter group. Variants (env):
  U_LR_A    lr of the added-row output layer (default: U_LR, i.e. the thesis setting)
  U_LR      lr of everything else: hidden layers, physical-row output, combinations, encoder (default 1e-5)
  U_INIT    'zero' (thesis) | 'hoekstra': linear skip on the added rows, x_a+ += W_lin z with W_lin random
            (Hoekstra 2026 Sec. 5.4.3: matrices not needed for baseline behaviour are random), physical
            rows of the skip absent, so the model equals the baseline at start (the readout is still zero)
            'identity': the same learnable skip initialised to the identity on the x_a columns and zero
            elsewhere (residual / discretised-ODE form x_a+ = x_a + f; Jan's identity_init_simple_res_net)
  U_SKIP_SCALE  scale of the random skip (default 1: torch default Linear init U(+-1/sqrt(n_in)))
Logged every EVAL updates on 72 fixed windows (encoder states of the first 24): fixed-window loss; added
self-map eigenvalues (d x_a+ / d x_a); full one-step Jacobian eigenvalues (8x8) and the share of points
with a complex pair in 95 to 380 Hz, its median frequency and zeta. Nothing outside outputs/.

Usage: python u_probe.py N_UPDATES BATCH TAG
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from e_thesis_checkpoints import cfg_for                                      # noqa: E402  (sets sys.path)
import n_phase1 as n1                                                          # noqa: E402

FS = 4000.0
EVAL = int(os.environ.get('U_EVAL', 25))


class RowSplitNet(torch.nn.Module):
    """Jan's zero-init MLP with its final layer split by rows (same function, same init)."""

    def __init__(self, net, n_phys, init, skip_scale):
        super().__init__()
        layers = list(net.net)
        last = layers[-1]
        self.hidden = torch.nn.Sequential(*layers[:-1])
        n_h, n_out = last.in_features, last.out_features
        self.out_p = torch.nn.Linear(n_h, n_phys).to(last.weight.dtype)
        self.out_a = torch.nn.Linear(n_h, n_out - n_phys).to(last.weight.dtype)
        for l in (self.out_p, self.out_a):
            torch.nn.init.zeros_(l.weight); torch.nn.init.zeros_(l.bias)
        self.skip = None
        self.polar = None
        if init == 'polar':
            # LRU-style complex pair on the x_a self-map (Orvieto et al. 2023 Sec. 3.3): A = r R(theta),
            # r = exp(-exp(nu)); drive from (x, u) through a zero-init linear part; x_a columns of the skip
            # unused. Start (r0, f0) from U_POLAR="r0,f0_Hz" (diagnostic starts, not a method).
            r0, f0 = (float(v) for v in os.environ.get('U_POLAR', '0.98,100').split(','))
            dt = last.weight.dtype
            self.nu = torch.nn.Parameter(torch.tensor(np.log(-np.log(r0)), dtype=dt))
            self.theta = torch.nn.Parameter(torch.tensor(2 * np.pi * f0 / FS, dtype=dt))
            self.skip = torch.nn.Linear(layers[0].in_features, n_out - n_phys, bias=False).to(dt)
            with torch.no_grad():
                self.skip.weight.zero_()
            self.polar = (n_phys, n_out)
        elif init in ('hoekstra', 'identity'):
            self.skip = torch.nn.Linear(layers[0].in_features, n_out - n_phys, bias=False).to(last.weight.dtype)
            with torch.no_grad():
                if init == 'hoekstra':
                    self.skip.weight.mul_(skip_scale)
                else:                     # residual form x_a+ = x_a + ...: identity on the x_a columns
                    self.skip.weight.zero_()
                    self.skip.weight[:, n_phys:n_out] = torch.eye(n_out - n_phys, dtype=last.weight.dtype)

    def forward(self, z):
        h = self.hidden(z)
        a = self.out_a(h)
        if self.skip is not None:
            a = a + self.skip(z)
        if self.polar is not None:
            p0, p1 = self.polar
            r = torch.exp(-torch.exp(self.nu)); c, s = torch.cos(self.theta), torch.sin(self.theta)
            xa = z[:, p0:p1]
            a = a + r * torch.stack([c * xa[:, 0] - s * xa[:, 1], s * xa[:, 0] + c * xa[:, 1]], 1)
        return torch.cat([self.out_p(h), a], 1)


def setup():
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    cfg = cfg_for('u', 2, 1)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg); norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files,
                                     val_files=data.val_files, val_data=data.val_ckpt_data, verbose=False)
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    old = {id(p) for p in ann.net.parameters()}
    torch.manual_seed(cfg.seed + 1000)
    ann.net = RowSplitNet(ann.net, 6, os.environ.get('U_INIT', 'zero'), float(os.environ.get('U_SKIP_SCALE', 1)))
    others = [p for g in fs.optimizer.param_groups for p in g['params'] if id(p) not in old]
    return cfg, data, norm, fs, ann, others


def jac_eigs(fs, x, u):
    """Full one-step Jacobian d x+ / d x (B, 8, 8) at states x, inputs u; its eigenvalues and those of the
    added self-map block."""
    nx = x.shape[1]
    f = lambda xx: fs.hfn(xx, u)[1].reshape(x.shape[0], -1)[:, :nx].sum(0)       # noqa: E731
    J = torch.autograd.functional.jacobian(f, x.clone()).permute(1, 0, 2).detach().numpy()
    return np.array([np.linalg.eigvals(j) for j in J]), np.array([np.linalg.eigvals(j[6:, 6:]) for j in J])


def band_pair(E, f_lo=95.0, f_hi=380.0):
    """Per point: the least-damped complex eigenvalue with frequency in [f_lo, f_hi] (freq, zeta) or nan."""
    out = np.full((len(E), 2), np.nan)
    for i, e in enumerate(E):
        e = e[e.imag > 1e-9]
        if not len(e):
            continue
        s = np.log(e) * FS
        f = np.abs(s.imag) / (2 * np.pi); z = -s.real / np.abs(s)
        m = (f >= f_lo) & (f <= f_hi)
        if m.any():
            j = np.argmin(np.where(m, z, np.inf)); out[i] = f[j], z[j]
    return out


def main(n_upd, B, tag):
    lr = float(os.environ.get('U_LR', 1e-5)); lr_a = float(os.environ.get('U_LR_A', lr))
    cfg, data, norm, fs, ann, others = setup()
    net = ann.net
    g_a = list(net.out_a.parameters()) + (list(net.skip.parameters()) if net.skip is not None else [])
    if net.polar is not None:
        g_a += [net.nu, net.theta]
        mask = torch.ones_like(net.skip.weight); mask[:, net.polar[0]:net.polar[1]] = 0
        net.skip.weight.register_hook(lambda g: g * mask)      # the self-map is the polar pair only
    # U_LR_H: lr of the shared hidden layers (default: U_LR); the physical-row output layer stays at U_LR
    lr_h = float(os.environ.get('U_LR_H', lr))
    opt = torch.optim.Adam([{'params': others, 'lr': lr}, {'params': list(net.out_p.parameters()), 'lr': lr},
                            {'params': list(net.hidden.parameters()), 'lr': lr_h},
                            {'params': g_a, 'lr': lr_a}], lr=lr, eps=cfg.adam_eps)
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    rng = np.random.default_rng(0)
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    print('\n######## probe [%s]: %d updates, batch %d, lr %g, lr added rows %g, init %s (skip scale %s), nf %d'
          % (tag, n_upd, B, lr, lr_a, os.environ.get('U_INIT', 'zero'), os.environ.get('U_SKIP_SCALE', 1), n1.NF),
          flush=True)
    hist = []

    def evaluate(it):
        with torch.no_grad():
            l = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
            x0 = fs.encoder(ev[0], ev[1])
        Ef, Ea = jac_eigs(fs, x0[:24], ev[2][:24, 0])
        rho = np.abs(Ea).max(1)
        dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
        fa = np.abs(np.angle(dom)) * FS / (2 * np.pi)
        pa, pf = band_pair(Ea), band_pair(Ef)
        share_a = np.mean(~np.isnan(pa[:, 0])); share_f = np.mean(~np.isnan(pf[:, 0]))
        print('  [eval %5d] loss %.4e | added: rho med %.3f max %.3f, dom freq med %.0f Hz, complex %.0f %%, '
              'pair in 95-380 Hz %.0f %% (f %.0f Hz zeta %.2f) | full J: pair in band %.0f %% (f %.0f Hz zeta %.2f)'
              ' | |W_a| %.2e' % (
                  it, l, np.median(rho), rho.max(), np.median(fa), 100 * np.mean(np.abs(Ea.imag).max(1) > 1e-9),
                  100 * share_a, np.nanmedian(pa[:, 0]) if share_a else np.nan, np.nanmedian(pa[:, 1]) if share_a else np.nan,
                  100 * share_f, np.nanmedian(pf[:, 0]) if share_f else np.nan, np.nanmedian(pf[:, 1]) if share_f else np.nan,
                  net.out_a.weight.norm().item()), flush=True)
        if net.polar is not None:
            print('      polar pair: r %.4f, f %.1f Hz' % (float(torch.exp(-torch.exp(net.nu))),
                                                       float(net.theta) * FS / (2 * np.pi)), flush=True)
        hist.append((it, l, np.median(rho), share_a, share_f))
        np.save(os.path.join(HERE, 'outputs', 'u_%s_hist.npy' % tag), np.array(hist))
        if not np.isfinite(l):
            raise SystemExit('non-finite loss, stopping')

    mh = [int(v) for v in os.environ['U_MH'].split(',')] if os.environ.get('U_MH') else None
    print('  multi-horizon: %s' % mh, flush=True)
    evaluate(0)
    t0 = time.time()
    for it in range(1, n_upd + 1):
        uh, yh, uf, yf, cix = n1.batch(recs, fs, na_r, nb_r, B, rng, cfg.dtype_pt)
        opt.zero_grad()
        if mh:
            # Multi-horizon objective (supervisor suggestion): the same windows truncated to each nf in
            # U_MH, weighted short first, long last. # HEURISTIC: tent weights over training progress
            # t = it / n_upd, centred at 0, 0.5, 1 for three horizons (no literature schedule; the
            # cited curricula use hard switches), normalised to sum 1.
            t = it / n_upd; c = np.linspace(0, 1, len(mh))
            w = np.maximum(0.0, 1 - np.abs(t - c) * (len(mh) - 1)); w = w / w.sum()
            loss = sum(wi * fs.loss(uh, yh, uf[:, :h], yf[:, :h], ctrl_ix=cix, nf=h)
                       for wi, h in zip(w, mh) if wi > 0)
        else:
            loss = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=n1.NF)
        loss.backward()
        if it <= 3 or it % 25 == 0:
            print('  upd %5d loss %.4e | %.2f s/upd' % (it, float(loss), (time.time() - t0) / it), flush=True)
        opt.step()
        if it % EVAL == 0 or it == n_upd:
            evaluate(it)
    torch.save({'hfn': fs.hfn.state_dict(), 'enc': fs.encoder.state_dict(), 'n_upd': n_upd},
               os.path.join(HERE, 'outputs', 'u_%s.pt' % tag))
    print('  saved outputs/u_%s.pt' % tag, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
