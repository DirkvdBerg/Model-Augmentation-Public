"""MIMO common-pole vector fitting, to replace the ill-conditioned monomial SK fit.

WHY THIS EXISTS. `frf_to_ss.py` fits a common-denominator model in a MONOMIAL basis, powers of s
up to 6 over 1 to 200 Hz, which spans ~14 decades in the regressor and is catastrophically
ill-conditioned. Two measured symptoms: accuracy is NOT monotone in line count (60 lines/decade
works, 120 and 240 fail), and `ident_check_lowf.py` showed the fit gets WORSE on the two real
poles when given MORE exact information. A Cramer-Rao bound (`crb_excitation.py`) then showed the
information is present in every band tested, so the deficiency is the basis, not the data.

`THEORY` Gustavsen, B., Semlyen, A., "Rational approximation of frequency domain responses by
vector fitting", IEEE Transactions on Power Delivery 14(3):1052-1061, 1999. Vector fitting
replaces the monomial basis with a PARTIAL-FRACTION basis on a set of starting poles, and
relocates the poles iteratively. The basis functions 1/(s - a_n) stay O(1) across the whole band
instead of spanning decades, which is the entire point. Same cure, and same reason, as the
bi-orthonormal polynomial basis of van Herpen et al., IEEE TAC 2015 (`references.md`); vector
fitting is the more widely used form and is what is implemented here.

Method, one iteration:
    solve   sum_n c_n/(s-a_n) + d - f(s) * sum_n ct_n/(s-a_n) = f(s)      (linear in c, d, ct)
    then    new poles = eig(A - b ct^T),   A = diag(a),  b = 1
because the zeros of sigma(s) = 1 + sum_n ct_n/(s-a_n) are the new poles. Repeat.

MIMO with COMMON poles: every element of the 3x3 shares `a_n` and `ct_n`, and has its own `c_n`
and `d`. That is the physically correct structure, since all channels of one system share a
denominator.

REAL-ONLY FORMULATION. Conjugate pole pairs are handled with the real basis pair
    phi1 = 1/(s-a) + 1/(s-a*),      phi2 = j/(s-a) - j/(s-a*)
so every unknown is real and the recovered poles are exactly conjugate rather than approximately.
Gustavsen's appendix; without it the fit can return near-conjugate pairs that break the
state-space realisation.

RIGID BODY. The plant has a double pole at s = 0 and vector fitting cannot place poles on the
imaginary axis. Following Voorhoeve et al. (IEEE TCST 29(1):194-206, 2021), the rigid-body
dynamics are FACTORED OUT rather than fitted: we fit H(s) = s^2 G(s), which is proper and has the
remaining 6 poles, and the full pole set is {0, 0} union poles(H).
"""
__project_origin__ = "added"

import numpy as np


def _basis(s, poles):
    """Real-parameterised partial-fraction basis.

    Returns Phi with columns ordered to match `_unpack`: one column per real pole, two per
    conjugate pair. Phi is complex (evaluated on the imaginary axis) but its COEFFICIENTS are
    real, which is what keeps the recovered poles exactly conjugate.
    """
    cols, kinds = [], []
    n = 0
    while n < len(poles):
        p = poles[n]
        if abs(p.imag) < 1e-12:
            cols.append(1.0 / (s - p.real))
            kinds.append(('real', n))
            n += 1
        else:
            q = poles[n + 1]                       # its conjugate, by construction
            cols.append(1.0 / (s - p) + 1.0 / (s - q))
            cols.append(1j / (s - p) - 1j / (s - q))
            kinds.append(('pair', n))
            n += 2
    return np.stack(cols, axis=1), kinds


def _relocate(poles, ct, kinds):
    """New poles = eigenvalues of (A - b ct^T) in the real block form."""
    N = len(poles)
    A = np.zeros((N, N))
    b = np.zeros(N)
    i = 0
    for kind, n in kinds:
        if kind == 'real':
            A[i, i] = poles[n].real
            b[i] = 1.0
            i += 1
        else:
            re, im = poles[n].real, poles[n].imag
            A[i, i] = re;      A[i, i + 1] = im
            A[i + 1, i] = -im; A[i + 1, i + 1] = re
            b[i] = 2.0; b[i + 1] = 0.0
            i += 2
    new = np.linalg.eigvals(A - np.outer(b, ct))
    return _canonical(new)


def _canonical(p, tol=1e-10):
    """Sort into [real..., pair, pair, ...] with each pair adjacent and +imag first."""
    real = np.sort(np.array([x.real for x in p if abs(x.imag) <= tol]))[::-1]
    cx = [x for x in p if x.imag > tol]
    cx.sort(key=lambda z: abs(z.imag))
    out = [complex(r, 0.0) for r in real]
    for z in cx:
        out += [z, np.conj(z)]
    return np.array(out)


def _enforce_stable(p):
    """Reflect any unstable pole into the left half plane. Standard in vector fitting."""
    q = p.copy()
    bad = q.real > 0
    q[bad] = -q[bad].real + 1j * q[bad].imag
    return q


def start_poles(f_lo, f_hi, n_real=2, n_pair=2):
    """Logarithmically spaced starting poles: `n_real` real, `n_pair` conjugate pairs.

    `HEURISTIC` Gustavsen's recipe is complex pairs spread over the band with
    Re = -Im/100. Two changes here, both from the plant rather than from the paper: some
    starting poles are REAL, because the plant has two real poles and starting every pole as a
    pair biases the relocation toward pairs; and the real ones start BELOW the band, because the
    two we are chasing sit below it. Vector fitting relocates poles freely, so these only set the
    basin, and `ident_check_lowf.py` re-run is what judges the choice.
    """
    w_lo, w_hi = 2 * np.pi * f_lo, 2 * np.pi * f_hi
    out = []
    for w in np.logspace(np.log10(w_lo / 10), np.log10(w_lo), n_real + 1)[:n_real]:
        out.append(complex(-w, 0.0))
    for w in np.logspace(np.log10(w_lo), np.log10(w_hi), n_pair + 2)[1:-1]:
        out += [complex(-w / 100, w), complex(-w / 100, -w)]
    return _canonical(np.array(out))


def vector_fit(f, G, poles=None, n_real=2, n_pair=2, iters=20, enforce_stable=True,
               weight='clipped', clip_pct=25.0, n_int=2):
    """Fit H(s) = s^2 G(s) with common poles. Returns (poles_of_H, residues, d, rms_rel).

    f : (K,) Hz.  G : (K, ny, nu) the FRF of the FULL plant, rigid body included.
    """
    s = 2j * np.pi * f
    # `n_int` is how many poles sit at s = 0. For THIS plant it is 2, but as TWO SIMPLE poles
    # rather than a Jordan block: `plant._K4` has zero rows for X and Y, so each free axis gives
    # {0, -c/m} independently (X: 0 and -0.647, Y: 0 and -0.990). Two simple poles at the same
    # location combine into ONE 1/s term whose residue matrix has rank 2, so the correct factor
    # to remove is s^1, not s^2. Using s^2 forces a Jordan block that the plant does not have,
    # which is what made the rank-1 factorisation fail by a factor of 7 while the residue fit
    # itself was accurate to 0.3 %.
    H = (s ** n_int)[:, None, None] * G
    K, ny, nu = H.shape
    Hf = H.reshape(K, ny * nu)
    m = ny * nu
    if poles is None:
        poles = start_poles(f[0], f[-1], n_real, n_pair)

    # `HEURISTIC` per-line, per-element RELATIVE weighting. Without it the fit minimises ABSOLUTE
    # error on H = s^2 G, and |s^2| grows as f^2, so 200 Hz outweighs 1 Hz by ~4e4 and the low
    # band is effectively discarded. Measured on the real FRF: unweighted put the absorber at
    # 1.34e-02 where the relative-weighted monomial fit reached 9.08e-06. Same reasoning as the
    # weight in `frf_to_ss.py`, and the same caveat: the literature's per-line total-variance
    # weight degenerates to 0/0 in a noiseless simulation, so this substitution is ours.
    # CLIPPED, not pure. Pure 1/|H| blows up at the ANTIRESONANCES, where |H| is smallest and the
    # measurement is least reliable, so it hands the fit to the worst data. Measured: on the
    # 1.5 %-error arm pure relative weighting put the 5.12 Hz pair at 10.68 Hz (worst |dz|
    # 4.37e-02) against 2.96e-03 unweighted. `THEORY` Voorhoeve, de Rozario, Aangenent, Oomen,
    # IEEE TCST 29(1):194-206, 2021 use exactly this: an inverse-magnitude weighting CLIPPED at a
    # maximum. `HEURISTIC` the clip level: |H| is floored at its `clip_pct` percentile per
    # element, so the weight never exceeds that reciprocal.
    aH = np.abs(Hf)
    if weight == 'none':
        Wt = np.ones_like(aH)
    else:
        floor = np.percentile(aH, clip_pct, axis=0, keepdims=True)
        Wt = 1.0 / np.maximum(aH, floor)

    for _ in range(iters):
        Phi, kinds = _basis(s, poles)                # (K, N)
        N = Phi.shape[1]
        # unknowns: [c_(elem,n) for each elem] , [d_elem] , [ct_n]
        ncol = m * (N + 1) + N
        rows, rhs = [], []
        for e in range(m):
            A = np.zeros((K, ncol), complex)
            A[:, e * N:(e + 1) * N] = Phi
            A[:, m * N + e] = 1.0
            A[:, m * (N + 1):] = -Hf[:, e][:, None] * Phi
            w = Wt[:, e][:, None]
            rows.append(A * w)
            rhs.append(Hf[:, e] * Wt[:, e])
        Aall = np.vstack(rows)
        ball = np.concatenate(rhs)
        Ar = np.vstack([Aall.real, Aall.imag])
        br = np.concatenate([ball.real, ball.imag])
        # column scaling: the ct block and the c blocks have very different magnitudes
        sc = np.linalg.norm(Ar, axis=0)
        sc[sc == 0] = 1.0
        theta, *_ = np.linalg.lstsq(Ar / sc, br, rcond=None)
        theta = theta / sc
        ct = theta[m * (N + 1):]
        new = _relocate(poles, ct, kinds)
        if enforce_stable:
            new = _enforce_stable(new)
        if np.allclose(np.sort_complex(new), np.sort_complex(poles), rtol=1e-12, atol=1e-14):
            poles = new
            break
        poles = new

    # final residue solve at fixed poles
    Phi, kinds = _basis(s, poles)
    N = Phi.shape[1]
    C = np.zeros((m, N)); D = np.zeros(m)
    for e in range(m):
        A = np.hstack([Phi, np.ones((K, 1), complex)]) * Wt[:, e][:, None]
        Ar = np.vstack([A.real, A.imag])
        bw = Hf[:, e] * Wt[:, e]
        br = np.concatenate([bw.real, bw.imag])
        th, *_ = np.linalg.lstsq(Ar, br, rcond=None)
        C[e] = th[:N]; D[e] = th[N]
    Hfit = (Phi @ C.T) + D[None, :]
    rel = np.abs(Hfit - Hf) / np.maximum(np.abs(Hf), 1e-300)
    return poles, C, D, float(np.median(rel))


def full_poles(poles_H):
    """The plant's pole set: the two rigid-body poles plus the fitted ones."""
    return np.concatenate([np.zeros(2, complex), poles_H])
