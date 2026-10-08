"""Landscape cut at a u_probe end point (identity-skip model): exact loss on 216 fixed windows when the
linear part of the added self-map (skip x_a block, A) is changed, drive, readout and NN part fixed.
  rot:   A + s [[0,-1],[1,0]]   (antisymmetric push; complex pair once s exceeds the real split)
  polar: A replaced by r R(2 pi f Ts) (diagnostic, as L1/O1; frequencies are scan points, not a method)
Prints loss / L0 and the eigenvalues of the resulting skip block. Nothing trains.

Usage: python w_rot_scan.py TAG   (U_INIT as for the probe)
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u_probe as up                                                           # noqa: E402
import n_phase1 as n1                                                          # noqa: E402


def main(tag):
    cfg, data, norm, fs, ann, others = up.setup()
    st = torch.load(os.path.join(HERE, 'outputs', 'u_%s.pt' % tag), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 216, np.random.default_rng(321), cfg.dtype_pt)
    W = ann.net.skip.weight
    A0 = W.data[:, 6:8].clone()

    def loss_with(A):
        with torch.no_grad():
            W.data[:, 6:8] = A
            l = float(fs.loss(*ev[:4], ctrl_ix=ev[4], nf=n1.NF))
            W.data[:, 6:8] = A0
        return l

    L0 = loss_with(A0)
    print('\n######## %s: L0 %.4e, A0 eig %s' % (tag, L0, np.round(np.linalg.eigvals(A0.numpy()), 4)), flush=True)
    J = torch.tensor([[0., -1.], [1., 0.]], dtype=A0.dtype)
    for s in (-0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4):
        A = A0 + s * J
        e = np.linalg.eigvals(A.numpy()); e1 = e[np.argmax(e.imag)]
        print('  rot s %+.2f: loss/L0 %.3f | eig |z| %.3f, %.0f Hz' % (
            s, loss_with(A) / L0, abs(e1), abs(np.angle(e1)) * up.FS / (2 * np.pi)), flush=True)
    for r in (0.9, 0.95, 0.98):
        row = []
        for f in (0, 50, 100, 150, 212, 260, 320):
            th = 2 * np.pi * f / up.FS
            R = torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=A0.dtype)
            row.append('%d Hz %.3f' % (f, loss_with(r * R) / L0))
        print('  polar r %.2f: %s' % (r, ' | '.join(row)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
