"""AE1: statistics of random live initialisations of the added states (no training).

Section 5.4-compatible live ResNet interpretation (Hoekstra et al. 2026; D-233): learning function = W_lin z + MLP(z),
MLP zero at init, W_lin rows into the 6 physical states zero, rows into the 2 added states random. At init the
physical states and the output equal the baseline, so the added states evolve as a driven linear system
    x_a(k+1) = W_aa x_a(k) + W_ax x(k) + W_au u(k),   z = [x (6), x_a (2), u (3)],
and their poles are eig(W_aa) (x is not affected by x_a at init: a cascade).
Two scales: U(-1, 1) (the paper's generic distribution for matrices not needed for baseline behaviour, p.10) and
U(-1/sqrt(11), 1/sqrt(11)) (torch's default Linear init, used by U2).
  Part 1 (closed form, N draws): unstable fraction (max |eig| > 1), complex fraction, distribution of the dominant
  radius and of the pair frequency, share with a pair in 95 to 380 Hz and in 150 to 300 Hz with |z| >= 0.8.
  Part 2 (data, first N2 draws per scale): x_a driven over 400 steps by the encoder's physical-state estimates and the
  inputs of 72 fixed training windows (a proxy for the baseline rollout's states), from the encoder's x_a(0):
  share of draws whose peak |x_a| stays below 1e3 (finite, usable rollout) and the median peak.

Usage: python ae_draw_stats.py [N] [N2]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FS = 4000.0
NX, NA, NU = 6, 2, 3
NZ = NX + NA + NU


def pole_stats(Waa):
    e = np.linalg.eigvals(Waa)                                   # (N, 2)
    rho = np.abs(e).max(1)
    cplx = np.abs(e.imag).max(1) > 1e-12
    ec = e[np.arange(len(e)), np.abs(e).argmax(1)]
    f = np.abs(np.angle(ec)) * FS / (2 * np.pi)
    s = np.log(ec.astype(complex)) * FS
    zeta = -s.real / np.maximum(np.abs(s), 1e-12)
    return rho, cplx, f, zeta


def main(N, N2):
    rng = np.random.default_rng(0)
    scales = (('U(-1,1)', 1.0), ('U(+-1/sqrt(11)), torch default', 1 / np.sqrt(NZ)))
    draws = {}
    print('\n######## AE1 live-init draw statistics: %d draws per scale (part 1), %d driven (part 2)' % (N, N2), flush=True)
    for name, b in scales:
        W = rng.uniform(-b, b, size=(N, NA, NZ))
        draws[name] = W
        rho, cplx, f, zeta = pole_stats(W[:, :, NX:NX + NA])
        stab = rho < 1
        band = cplx & (f >= 95) & (f <= 380)
        near = cplx & (f >= 150) & (f <= 300) & (rho >= 0.8) & stab
        print('  [%s] unstable %.1f %% | complex %.1f %% | dominant |z| quantiles 10/50/90 %% %.2f / %.2f / %.2f'
              ' | complex-pair frequency quantiles 10/50/90 %% %.0f / %.0f / %.0f Hz' % (
                  name, 100 * np.mean(~stab), 100 * np.mean(cplx), *np.quantile(rho, [.1, .5, .9]),
                  *(np.quantile(f[cplx], [.1, .5, .9]) if cplx.any() else (np.nan,) * 3)), flush=True)
        print('      pair in 95-380 Hz: %.2f %% of draws (stable %.2f %%); pair in 150-300 Hz with 0.8 <= |z| < 1: %.3f %%;'
              ' median zeta of in-band stable pairs %.2f' % (
                  100 * np.mean(band), 100 * np.mean(band & stab), 100 * np.mean(near),
                  np.median(zeta[band & stab]) if (band & stab).any() else np.nan), flush=True)
    # part 2: driven by data
    import u_probe as up
    import n_phase1 as n1
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    na, nb = fs.na, fs.nb
    X, U, XA0 = [], [], []
    for r, k in [(i % len(recs), recs[i % len(recs)][2][(37 * i) % (len(recs[i % len(recs)][2]) - 2)]) for i in range(72)]:
        u, y, st, row = recs[r]
        ks = np.arange(k, k + n1.NF)
        uh = np.stack([u[j - nb:j + nb_r] for j in ks]); yh = np.stack([y[j - na:j + na_r] for j in ks])
        with torch.no_grad():
            xe = fs.encoder(torch.as_tensor(np.ascontiguousarray(uh), dtype=cfg.dtype_pt),
                            torch.as_tensor(np.ascontiguousarray(yh), dtype=cfg.dtype_pt)).numpy()
        X.append(xe[:, :NX]); U.append(u[ks]); XA0.append(xe[0, NX:])
    X, U, XA0 = np.stack(X), np.stack(U), np.stack(XA0)          # (72, 400, 6), (72, 400, 3), (72, 2)
    print('  part 2 inputs: 72 windows x %d steps; encoder |x| rms %.2f, |u| rms %.2f, x_a(0) rms %.2f' % (
        n1.NF, np.sqrt((X ** 2).mean()), np.sqrt((U ** 2).mean()), np.sqrt((XA0 ** 2).mean())), flush=True)
    for name, _ in scales:
        W = draws[name][:N2]
        peaks = np.empty(len(W))
        with np.errstate(over='ignore', invalid='ignore'):
            for i, w in enumerate(W):
                Waa, Wax, Wau = w[:, NX:NX + NA], w[:, :NX], w[:, NX + NA:]
                xa = XA0.copy(); pk = np.abs(xa).max()
                drive = X @ Wax.T + U @ Wau.T                       # (72, 400, 2)
                for t in range(n1.NF):
                    xa = xa @ Waa.T + drive[:, t]
                    pk = max(pk, np.abs(xa).max())
                    if not np.isfinite(pk) or pk > 1e300:
                        pk = np.inf; break
                peaks[i] = pk
        rho = np.abs(np.linalg.eigvals(W[:, :, NX:NX + NA])).max(1)
        ok = peaks < 1e3
        print('  [%s] driven 400 steps: peak |x_a| < 1e3 in %.1f %% of draws (stable-pole draws %.1f %%, of which %.1f %%'
              ' below 1e3); median peak %.3g, stable-draw median peak %.3g' % (
                  name, 100 * ok.mean(), 100 * np.mean(rho < 1), 100 * ok[rho < 1].mean() if (rho < 1).any() else np.nan,
                  np.median(peaks), np.median(peaks[rho < 1]) if (rho < 1).any() else np.nan), flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100000, int(sys.argv[2]) if len(sys.argv) > 2 else 500)
