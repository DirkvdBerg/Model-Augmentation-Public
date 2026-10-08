"""Does the barrier against moving the added poles depend on the horizon? Evaluation only (no training).

Run 42 (unchanged) with the y_valley measuring probe Delta x_a on the added rows, Delta(s) = s (A_T - A0),
A_T = 0.95 R(212 Hz) (diagnostic destination). Drive, readout and everything else FIXED (the uphill cut of
W1 / O1). Exact loss on 216 fixed windows truncated to nf in NFS (same window starts), normalised by its own
s = 0 value. A barrier that shrinks with nf (or a cut that turns downhill at short nf) supports a
short-to-long horizon schedule as the in-framework remedy.

Usage: python z_horizon_cut.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import h_learned_poles as hp_                                                  # noqa: E402
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402
from y_valley import TrainedSplit                                              # noqa: E402

NFS = (25, 50, 100, 200, 400)
S = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def main():
    from model_augmentation.fit_systems.blocks import Static_ANN_Block
    cfg, data, norm, fs = hp_.build('86894')
    ann = next(b for b in fs.hfn.connected_blocks if isinstance(b, Static_ANN_Block))
    ann.net = TrainedSplit(ann.net, 6)
    net = ann.net
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 216, np.random.default_rng(321), cfg.dtype_pt)
    with torch.no_grad():
        x0 = fs.encoder(ev[0], ev[1])[:24]
    u0 = ev[2][:24, 0]
    f = lambda xx: fs.hfn(xx, u0)[1].reshape(x0.shape[0], -1)[:, 6:8].sum(0)          # noqa: E731
    A0 = torch.autograd.functional.jacobian(f, x0.clone()).permute(1, 0, 2)[:, :, 6:8].detach().mean(0)
    th = 2 * np.pi * 212 / up.FS
    A_T = 0.95 * torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=A0.dtype)
    print('\n######## horizon cut, run 42, drive/readout fixed; loss(s) / loss(0) per nf', flush=True)
    for nf in NFS:
        row = []
        for s in S:
            with torch.no_grad():
                net.delta.copy_(s * (A_T - A0))
                l = float(fs.loss(ev[0], ev[1], ev[2][:, :nf], ev[3][:, :nf], ctrl_ix=ev[4], nf=nf))
            row.append(l)
        row = np.array(row)
        print('  nf %3d (%.1f ms): L0 %.4e | %s' % (nf, nf / 4.0, row[0], ' '.join(
            's%.2f %.3f' % (s, v) for s, v in zip(S, row / row[0]))), flush=True)
    net.delta.zero_()


if __name__ == '__main__':
    main()
