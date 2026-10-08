"""Both added eigenvalues and the full one-step Jacobian spectrum of a saved u_probe checkpoint, at the
encoder states of the 72 fixed eval windows. Set U_INIT as for the probe that wrote the checkpoint.

Usage: python v_inspect.py TAG
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import u_probe as up                                                           # noqa: E402
import n_phase1 as n1                                                          # noqa: E402


def hz_zeta(e):
    s = np.log(e.astype(complex)) * up.FS
    return np.abs(s.imag) / (2 * np.pi), -s.real / np.abs(s)


def main(tag):
    cfg, data, norm, fs, ann, others = up.setup()
    st = torch.load(os.path.join(HERE, 'outputs', 'u_%s.pt' % tag), weights_only=False)
    fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    ev = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    with torch.no_grad():
        x0 = fs.encoder(ev[0], ev[1])
    Ef, Ea = up.jac_eigs(fs, x0, ev[2][:, 0])
    Ea = np.sort_complex(Ea)
    print('\n######## %s (%d updates): added self-map eigenvalues at %d points' % (tag, st['n_upd'], len(Ea)))
    for q, name in ((0, 'min'), (50, 'median'), (100, 'max')):
        print('  %-6s eig1 %s  eig2 %s' % (name, np.percentile(Ea[:, 0].real, q), np.percentile(Ea[:, 1].real, q)))
    print('  imag max %.2e' % np.abs(Ea.imag).max())
    sk = ann.net.skip.weight.detach().numpy() if ann.net.skip is not None else None
    if sk is not None:
        print('  skip x_a block (linear part of the self-map):\n', sk[:, 6:8], '\n  eig', np.linalg.eigvals(sk[:, 6:8]))
    f, z = hz_zeta(Ef.reshape(-1))
    m = np.abs(Ef.reshape(-1).imag) > 1e-9
    print('  full J: complex eigenvalues f/zeta (all points pooled), median per 50 Hz bin:')
    for lo in range(0, 2000, 50):
        k = m & (f >= lo) & (f < lo + 50)
        if k.any():
            print('    %4d-%4d Hz: n %4d  |z| med %.3f  zeta med %.3f' % (
                lo, lo + 50, k.sum(), np.median(np.abs(Ef.reshape(-1)[k])), np.median(z[k])))
    print('  full J |eig| sorted, median over points: %s' % np.round(np.median(np.sort(np.abs(Ef), 1), 0), 4))
    np.savez(os.path.join(HERE, 'outputs', 'v_%s.npz' % tag), Ef=Ef, Ea=Ea)


if __name__ == '__main__':
    main(sys.argv[1])
