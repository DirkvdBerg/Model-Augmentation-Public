"""AA1: what do the opening TARGETS ask for, at their optimum? (diagnostic, nothing trains)

On a probe checkpoint (u_probe / f_eeoe format, zero-init RowSplitNet model) and contiguous stretches of the
training records (encoder states at every consecutive k):
  1. Encoder x_a content: lag-1 autocorrelation, power fractions per band (Welch, 4 kHz).
  2. Equation-error ceiling: the LINEAR least-squares optimum of the EE term on the added rows,
     x_a(k+1) ~ A x_a(k) + Bx x(k) + Bu u(k) + c, eigenvalues of A (the best any F can do to first order),
     vs an instrumental-variable (IV) estimate with instruments [x_hat(k-d), u(k-d)], d beyond the encoder
     window, so white measurement noise in the regressor history is uncorrelated with the instruments.
     # THEORY: errors-in-variables attenuation of LS and its IV correction (Soderstrom and Stoica 1983,
     # Instrumental Variable Methods for System Identification, basic IV estimator).
     Same two fits on the physical rows as a sanity check (rigid modes near z = 1 expected).
  3. Model's own A_aa (Jacobian) eigenvalues at the same points.
  4. Innovation target optimum: the current model's next M output errors from every start k (closed-loop
     rollout, M steps), normalised as in E3; reduced-rank (rank 2) linear regression on the encoder's input
     history [u_hist, y_hist] = the best 2-d linear encoder state for the head; explained fraction (rank 2 and
     full rank); then LS and IV one-step dynamics of that optimal 2-d state (its poles = what EE would chase
     if the encoder reached the head's optimum).
  5. Spectrum of the 400-step rollout error on 72 fixed windows (band fractions).

Usage: python aa_target_ceiling.py CKPT_FILE [M]     (file in outputs/; 'none' = the zero-init start)
"""
import os
import sys

import numpy as np
import torch
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n_phase1 as n1                                                          # noqa: E402
import u_probe as up                                                           # noqa: E402

FS = 4000.0
BANDS = [(0, 50), (50, 150), (150, 300), (300, 800), (800, 2000)]
N_SEG, N_REC = 3000, 4          # contiguous starts per record, records used
D_IV = (20, 40)                 # instrument delays (encoder window 17 samples)


def band_frac(x):
    f, p = welch(x, fs=FS, nperseg=512, axis=0)
    p = p.reshape(len(f), -1).sum(1)
    return ' '.join('%d-%d:%.2f' % (a, b, p[(f >= a) & (f < b)].sum() / p.sum()) for a, b in BANDS)


def pole_str(A):
    e = np.linalg.eigvals(A)
    s = np.log(e.astype(complex)) * FS
    return ', '.join('%.3f@%.0fHz(z%.2f)' % (abs(v), abs(si.imag) / (2 * np.pi), -si.real / max(abs(si), 1e-12))
                     for v, si in zip(e, s))


def fit(Y, R, Z=None):
    """LS (Z None) or basic IV estimate of Y ~ R Theta (Y (N, p), R (N, q), instruments Z (N, q))."""
    if Z is None:
        return np.linalg.lstsq(R, Y, rcond=None)[0]
    return np.linalg.lstsq(Z.T @ R, Z.T @ Y, rcond=None)[0]


def dyn_report(name, S, X, U, idx_self):
    """One-step LS and IV fits S(k+1) ~ [S(k), X(k), U(k), 1]; S is (N, p), stacked per record.
    idx_self: columns of the regressor that are S itself (block whose eigenvalues are reported)."""
    out = []
    for d in (None,) + D_IV:
        Ys, Rs, Zs = [], [], []
        for s, x, u in zip(S, X, U):
            k = np.arange(max(d or 0, 0), len(s) - 1)
            R = np.c_[s[k], x[k], u[k], np.ones(len(k))]
            Ys.append(s[k + 1]); Rs.append(R)
            if d is not None:
                Zs.append(np.c_[s[k - d], x[k - d], u[k - d], np.ones(len(k))])
        Y, R = np.concatenate(Ys), np.concatenate(Rs)
        Th = fit(Y, R, None if d is None else np.concatenate(Zs))
        A = Th[idx_self].T
        res = Y - R @ Th
        out.append('    %s %s: poles %s | residual/var %.3f' % (
            name, 'LS' if d is None else 'IV d=%d' % d, pole_str(A), res.var(0).sum() / Y.var(0).sum()))
    print('\n'.join(out), flush=True)


def main(fname, M):
    os.environ['U_INIT'] = 'zero'
    cfg, data, norm, fs, ann, others = up.setup()
    if fname != 'none':
        st = torch.load(os.path.join(HERE, 'outputs', fname), weights_only=False)
        fs.hfn.load_state_dict(st['hfn']); fs.encoder.load_state_dict(st['enc'])
    recs, na_r, nb_r = n1.windows(cfg, data, fs)
    print('\n######## AA1 target ceiling [%s], M %d, encoder na %d nb %d (+%d right)' % (fname, M, fs.na, fs.nb, na_r),
          flush=True)
    order = np.argsort([-len(r[2]) for r in recs])[:N_REC]
    XS, XA, US, HS, ES = [], [], [], [], []
    for r in order:
        u, y, starts, row = recs[r]
        ks = np.arange(starts[0], starts[0] + N_SEG)
        uh = np.stack([u[k - fs.nb:k + nb_r] for k in ks]); yh = np.stack([y[k - fs.na:k + na_r] for k in ks])
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=cfg.dtype_pt)   # noqa: E731
        with torch.no_grad():
            x = fs.encoder(T(uh), T(yh)).numpy()
            uf = np.stack([u[k:k + M] for k in ks]); yf = np.stack([y[k:k + M] for k in ks])
            cix = torch.full((len(ks),), row, dtype=torch.long)
            yp, _ = fs.simulate(T(x), T(uf), T(yf), ctrl_ix=cix, nf=M)
        XS.append(x[:, :6]); XA.append(x[:, 6:]); US.append(u[ks])
        HS.append(np.c_[uh.reshape(len(ks), -1), yh.reshape(len(ks), -1)])
        ES.append((yf - yp.numpy()).reshape(len(ks), -1))
    xa = np.concatenate(XA)
    ac = [np.corrcoef(a[1:, j], a[:-1, j])[0, 1] for a in XA for j in range(2)]
    print('  1. encoder x_a: rms %s, lag-1 autocorr per record/coord %s' % (
        np.round(xa.std(0), 3), np.round(ac, 3)), flush=True)
    print('     x_a band power fractions: %s' % band_frac(np.concatenate([a - a.mean(0) for a in XA])), flush=True)
    print('     x_phys band power fractions: %s' % band_frac(np.concatenate([a - a.mean(0) for a in XS])), flush=True)
    print('  2. one-step dynamics of the ENCODER states (EE ceiling):', flush=True)
    dyn_report('x_a', XA, XS, US, slice(0, 2))
    dyn_report('x_phys', XS, XA, US, slice(0, 6))
    # 3. model Jacobian at a subset of the same points
    xall = torch.as_tensor(np.c_[np.concatenate(XS), xa][::250], dtype=cfg.dtype_pt)
    uall = torch.as_tensor(np.concatenate(US)[::250], dtype=cfg.dtype_pt)
    Ef, Ea = up.jac_eigs(fs, xall, uall)
    dom = Ea[np.arange(len(Ea)), np.abs(Ea).argmax(1)]
    print('  3. model A_aa: dominant |z| median %.3f, complex %.0f %%, median |angle| %.0f Hz' % (
        np.median(np.abs(dom)), 100 * np.mean(np.abs(Ea.imag).max(1) > 1e-9),
        np.median(np.abs(np.angle(dom))) * FS / (2 * np.pi)), flush=True)
    # 4. innovation target optimum (normalised as E3: divide by the global RMS)
    H = np.concatenate(HS); E = np.concatenate(ES); E = E / np.sqrt((E ** 2).mean())
    Hc = np.c_[H, np.ones(len(H))]
    B = np.linalg.lstsq(Hc, E, rcond=None)[0]
    Ehat = Hc @ B
    U_, s_, Vt = np.linalg.svd(Ehat - Ehat.mean(0), full_matrices=False)
    ev = (E - E.mean(0)).var(0).sum()
    print('  4. innovation (M %d): rms per channel %s; linear-in-history explained: full rank %.3f, rank 2 %.3f, rank 4 %.3f'
          % (M, np.round(np.sqrt((ES[0].reshape(N_SEG, M, -1) ** 2).mean((0, 1))), 5),
             1 - (E - Ehat).var(0).sum() / ev,
             (s_[:2] ** 2).sum() / len(E) / ev, (s_[:4] ** 2).sum() / len(E) / ev), flush=True)
    print('     innovation first-sample band fractions: %s' % band_frac(
        np.concatenate([e.reshape(N_SEG, M, -1)[:, 0, :] for e in ES])), flush=True)
    Xs = Hc @ B @ Vt[:2].T                                  # optimal 2-d state (up to an invertible map)
    XSs = np.split(Xs, len(order))
    print('     optimal 2-d state: lag-1 autocorr %s; band fractions %s' % (
        np.round([np.corrcoef(a[1:, j], a[:-1, j])[0, 1] for a in XSs for j in range(2)], 3),
        band_frac(np.concatenate([a - a.mean(0) for a in XSs]))), flush=True)
    dyn_report('x_inn*', XSs, XS, US, slice(0, 2))
    # current encoder x_a vs that optimum: canonical correlation proxy (R^2 of x_inn* on x_a)
    A_ = np.c_[xa, np.ones(len(xa))]
    r2 = [1 - (Xs[:, j] - A_ @ np.linalg.lstsq(A_, Xs[:, j], rcond=None)[0]).var() / Xs[:, j].var() for j in range(2)]
    print('     R^2 of the optimal state on the current x_a: %s' % np.round(r2, 3), flush=True)
    # 5. rollout error spectrum on the fixed eval windows
    ev_ = n1.batch(recs, fs, na_r, nb_r, 72, np.random.default_rng(123), cfg.dtype_pt)
    with torch.no_grad():
        x0 = fs.encoder(ev_[0], ev_[1])
        yp, _ = fs.simulate(x0, ev_[2], ev_[3], ctrl_ix=ev_[4], nf=n1.NF)
    e = (ev_[3] - yp).numpy()
    print('  5. 400-step rollout error: loss %.4e; band fractions per channel:' % float((e ** 2).mean()), flush=True)
    for c in range(e.shape[2]):
        print('     ch %d: %s' % (c, band_frac(e[:, :, c].T)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
