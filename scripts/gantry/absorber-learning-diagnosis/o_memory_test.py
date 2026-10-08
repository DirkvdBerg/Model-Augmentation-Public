"""Is growing the added-state MEMORY downhill at the phase-1 point? (after n_phase1.py)

Loads a phase-1 state (outputs/n_phase1_<tag>.pt) and adds a linear self-feedback on the added
states, x_a+ = state(z) + Delta x_a (Delta 2x2, zero = the trained phase-1 model). Measured on fixed
windows (12 per training record, exact loss, encoder x0):
  1. the gradient dL/dDelta at Delta = 0, and the loss and added-state eigenvalues along the
     steepest-descent line Delta = -s g / |g|: does descent grow memory, and toward which frequency?
  2. the L1 analogue: Delta = r R(2 pi f Ts) (eigenvalues r e^{+-i 2 pi f Ts} in the current gauge)
     for r 0.3 to 0.95 and f 150 / 212 / 260 Hz. Gauge caveat: only one member of the similarity
     class of Delta is tried, so a rise here does not exclude every realisation; a fall is sufficient.
Reading: loss falling as r grows (at any f, deepest near 212 Hz) = the route from the phase-1 point
to a resonator is downhill (the two-phase training has a mechanism to work); loss rising = stalled.

Usage: python o_memory_test.py [tag]   (default lr1e-5)
"""
import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n_phase1 import setup, windows, added_eigs, NF                          # noqa: E402

TS = 1 / 4000.0
NPR = 12


def main(tag):
    cfg, data, norm, fs, ann, _ = setup()
    st = torch.load(os.path.join(HERE, 'outputs', 'n_phase1_%s.pt' % tag), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    net = ann.net
    net.delta = torch.zeros(2, 2, dtype=cfg.dtype_pt, requires_grad=True)
    base_fwd = type(net).forward

    def fwd(self, z):
        out = base_fwd(self, z)
        a = z[:, 6:8]
        return torch.cat([out[:, :6], out[:, 6:] + a @ self.delta.T], 1)
    net.forward = types.MethodType(fwd, net)

    recs, na_r, nb_r = windows(cfg, data, fs)
    W = []
    for (u, y, starts, row) in recs:
        pick = starts[np.linspace(0, len(starts) - 1, NPR).round().astype(int)]
        T = lambda a: torch.as_tensor(np.ascontiguousarray(np.stack(a)), dtype=cfg.dtype_pt)   # noqa: E731
        W.append((T([u[k - fs.nb:k + nb_r] for k in pick]), T([y[k - fs.na:k + na_r] for k in pick]),
                  T([u[k:k + NF] for k in pick]), T([y[k:k + NF] for k in pick]), torch.full((NPR,), row, dtype=torch.long)))

    def loss(grad=False):
        tot, n = 0.0, 0
        for uh, yh, uf, yf, c in W:
            if grad:
                l = fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=NF); (l * len(c)).backward(); tot += float(l) * len(c)
            else:
                with torch.no_grad():
                    tot += float(fs.loss(uh, yh, uf, yf, ctrl_ix=c, nf=NF)) * len(c)
            n += len(c)
        return tot / n

    uh, yh, uf = W[0][0], W[0][1], W[0][2]
    with torch.no_grad():
        x0 = fs.encoder(uh, yh)

    def eig_summary():
        e = added_eigs(fs, ann, x0, uf[:, 0])
        k = np.abs(e).argmax(1); m = e[np.arange(len(e)), k]
        return np.median(np.abs(m)), np.median(np.abs(np.angle(m))) / (2 * np.pi * TS)

    L0 = loss(grad=True)
    g = net.delta.grad.detach().clone()
    r0, f0 = eig_summary()
    print('\n######## memory test [%s]: %d windows; loss at Delta 0 %.4e; added eig rho %.4f at %.0f Hz' % (
        tag, NPR * len(W), L0, r0, f0), flush=True)
    print('  dL/dDelta = %s  (|g| %.3e; trace part %.3e = radial, antisym part %.3e = rotation)' % (
        np.array2string(g.numpy(), precision=3), float(g.norm()), float(torch.trace(g)) / 2, float(g[1, 0] - g[0, 1]) / 2), flush=True)
    d = -g / g.norm()
    print('  steepest-descent line Delta = s d:')
    for s in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
        with torch.no_grad():
            net.delta.copy_(s * d)
        r, f = eig_summary()
        print('    s %.2f: loss / L0 %.3f | added eig rho %.4f at %.0f Hz' % (s, loss() / L0, r, f), flush=True)
    print('  L1 analogue, Delta = r R(2 pi f Ts), loss / L0:')
    print('    r      ' + ' '.join('%8d Hz' % f for f in (150, 212, 260)))
    for r in (0.3, 0.6, 0.8, 0.9, 0.95):
        row = []
        for f in (150, 212, 260):
            th = 2 * np.pi * f * TS
            with torch.no_grad():
                net.delta.copy_(torch.tensor([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=cfg.dtype_pt) * r)
            row.append(loss() / L0)
        print('    %.2f   %s' % (r, ' '.join('%11.3f' % v for v in row)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'lr1e-5')
