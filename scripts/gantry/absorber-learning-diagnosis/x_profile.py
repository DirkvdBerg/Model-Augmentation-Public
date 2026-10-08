"""Profile test at a u_probe end point (identity-skip model; diagnostic only): the linear part of the added
self-map (skip x_a block) is FIXED at a chosen pole location, everything else (drive, readout, NN, physical
rows, combinations, encoder) re-adapts for N updates with the probe's settings, from the same start and on
the same batch sequence for every configuration. Compared with a control that keeps the learned block.
The band poles are scan points of a diagnostic (as L1), never a method.

Usage: python x_profile.py TAG N_UPDATES BATCH   (U_INIT and U_LR_A as for the probe)
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u_probe as up                                                           # noqa: E402
import n_phase1 as n1                                                          # noqa: E402

CONFIGS = [('control', None), ('r0.98_212Hz', (0.98, 212)), ('r0.95_212Hz', (0.95, 212)),
           ('r0.98_260Hz', (0.98, 260)), ('r0.95_150Hz', (0.95, 150))]


def main(tag, n_upd, B):
    lr = float(os.environ.get('U_LR', 1e-5)); lr_a = float(os.environ.get('U_LR_A', lr))
    cfg, data, norm, fs, ann, others = up.setup()
    st = torch.load(os.path.join(HERE, 'outputs', 'u_%s.pt' % tag), weights_only=False)
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 216, np.random.default_rng(321), cfg.dtype_pt)
    net = ann.net
    W = net.skip.weight
    mask = torch.ones_like(W); mask[:, 6:8] = 0
    W.register_hook(lambda g: g * mask)                       # x_a block of the skip frozen
    print('\n######## profile [%s]: %d updates, batch %d, lr %g, lr added %g' % (tag, n_upd, B, lr, lr_a), flush=True)
    for name, pole in CONFIGS:
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
        if pole is not None:
            r, f = pole; th = 2 * np.pi * f / up.FS
            with torch.no_grad():
                W[:, 6:8] = r * torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=W.dtype)
        g_a = list(net.out_a.parameters()) + [W]
        g_rest = list(net.hidden.parameters()) + list(net.out_p.parameters())
        opt = torch.optim.Adam([{'params': others, 'lr': lr}, {'params': g_rest, 'lr': lr},
                                {'params': g_a, 'lr': lr_a}], lr=lr, eps=cfg.adam_eps)
        rng = np.random.default_rng(7)
        with torch.no_grad():
            l_start = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
        t0 = time.time(); ls = []
        for it in range(1, n_upd + 1):
            uh, yh, uf, yf, cix = n1.batch(recs, fs, na_r, nb_r, B, rng, cfg.dtype_pt)
            opt.zero_grad()
            loss = fs.loss(uh, yh, uf, yf, ctrl_ix=cix, nf=n1.NF)
            loss.backward(); opt.step()
            if it % 50 == 0 or it == n_upd:
                with torch.no_grad():
                    ls.append(float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF)))
        e = np.linalg.eigvals(W.data[:, 6:8].numpy())
        print('  %-12s start %.4e -> %s | skip eig %s | %.0f s' % (
            name, l_start, ' -> '.join('%.4e' % v for v in ls), np.round(e, 4), time.time() - t0), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
