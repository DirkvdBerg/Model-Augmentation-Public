"""Shaping filter H_v for the source noise (CN-006): Yule-Walker AR / VAR from a spectral matrix.

    v_t = sum_{i=1..p} A_i v_{t-i} + L w_t,   w_t ~ N(0, I) per sample,   Sigma = L L^T.

THEORY: Yule-Walker on a positive definite block-Toeplitz autocovariance gives a causal, stable
AR model (Whittle 1963; Brockwell and Davis 1991 Sec. 11.3). Spectral conventions as in
measure/spec.py: one-sided density, P[f, i, j] = E[X_i X_j^*].
"""
import numpy as np
from scipy.signal import fftconvolve, lfilter


def autocov(P, fs, nfft=None):
    """One-sided spectral matrix P (F = nfft/2 + 1, m, m) -> R (nfft, m, m), R[k] = E[v_{t+k} v_t^T]."""
    F = P.shape[0]
    N = 2 * (F - 1) if nfft is None else nfft
    full = np.zeros((N,) + P.shape[1:], complex)
    full[0] = P[0]
    full[1:N // 2] = P[1:N // 2] / 2
    full[N // 2] = P[N // 2]
    full[N // 2 + 1:] = np.conj(full[1:N // 2][::-1])
    return np.real(np.fft.ifft(full, axis=0)) * fs


def fit_var(R, p, diagonal=False):
    """Yule-Walker. R (N, m, m). Returns A (p, m, m), Sigma (m, m)."""
    m = R.shape[1]
    if diagonal:
        A = np.zeros((p, m, m)); Sig = np.zeros((m, m))
        for i in range(m):
            a, s = fit_var(R[:, i:i + 1, i:i + 1], p)
            A[:, i, i] = a[:, 0, 0]; Sig[i, i] = s[0, 0]
        return A, Sig
    Rk = lambda k: R[k] if k >= 0 else R[-k].T                           # noqa: E731
    G = np.zeros((m * p, m * p))
    for i in range(p):
        for k in range(p):
            G[i * m:(i + 1) * m, k * m:(k + 1) * m] = Rk(k - i)
    Rv = np.hstack([Rk(k) for k in range(1, p + 1)])                    # (m, m p)
    Aflat = np.linalg.solve(G.T, Rv.T).T                                # Rv = A G
    A = np.stack([Aflat[:, i * m:(i + 1) * m] for i in range(p)])
    Sig = R[0] - sum(A[i] @ Rk(i + 1).T for i in range(p))
    Sig = (Sig + Sig.T) / 2
    return A, Sig


def model_psd(A, Sig, f, fs):
    """One-sided spectral matrix of the VAR at frequencies f: 2 Ts H Sigma H^H (DC not doubled)."""
    p, m, _ = A.shape
    z = np.exp(-2j * np.pi * np.asarray(f) / fs)
    Hinv = np.eye(m)[None] - np.einsum('kf,kij->fij', z[None, :] ** np.arange(1, p + 1)[:, None], A)
    H = np.linalg.inv(Hinv)
    P = 2.0 / fs * H @ Sig[None] @ np.conj(np.transpose(H, (0, 2, 1)))
    P[np.asarray(f) == 0] /= 2
    return P


def is_stable(A):
    p, m, _ = A.shape
    comp = np.zeros((m * p, m * p))
    comp[:m] = np.hstack(list(A))
    comp[m:, :-m] = np.eye(m * (p - 1))
    return float(np.abs(np.linalg.eigvals(comp)).max())


def impulse(A, n):
    """MA coefficients Psi_k (n, m, m) of (I - sum A_i z^-i)^-1."""
    p, m, _ = A.shape
    Psi = np.zeros((n, m, m)); Psi[0] = np.eye(m)
    for k in range(1, n):
        acc = np.zeros((m, m))
        for i in range(1, min(p, k) + 1):
            acc += A[i - 1] @ Psi[k - i]
        Psi[k] = acc
    return Psi


def is_diagonal(A, Sig):
    off = lambda M: M - np.einsum('...ii->...i', M)[..., None] * np.eye(M.shape[-1])  # noqa: E731
    return not (np.any(off(A)) or np.any(off(Sig)))


def generate(A, Sig, n, rng, burn=8192):
    """n samples (n, m) of the AR/VAR driven by white noise from `rng`; the first `burn` samples
    (start-up transient from a zero past) are discarded.

    Diagonal models use lfilter (exact recursion); a full VAR uses its MA form truncated at
    `burn` lags, by FFT convolution.
    """
    p, m, _ = A.shape
    L = np.linalg.cholesky(Sig)
    e = rng.standard_normal((n + burn, m)) @ L.T
    if is_diagonal(A, Sig):
        out = np.column_stack([lfilter([1.0], np.concatenate([[1.0], -A[:, i, i]]), e[:, i])
                               for i in range(m)])
        return out[burn:]
    Psi = impulse(A, burn)
    out = np.zeros((n + burn, m))
    for i in range(m):
        for j in range(m):
            out[:, i] += fftconvolve(e[:, j], Psi[:, i, j])[:n + burn]
    return out[burn:]


def impulse_fft(A, n=None, tol=1e-9, nmax=65536):
    """MA coefficients Psi (n, m, m) of (I - sum A_i z^-i)^-1 by FFT (CN-006a).

    n defaults to the lag where the slowest AR root has decayed below `tol` (capped at nmax).
    """
    p, m, _ = A.shape
    if n is None:
        rho = is_stable(A)
        n = int(min(nmax, max(4 * p, np.ceil(np.log(tol) / np.log(rho)))))
    N = 1 << int(np.ceil(np.log2(4 * max(n, p + 1))))
    B = np.zeros((N, m, m)); B[0] = np.eye(m); B[1:p + 1] = -A
    H = np.linalg.inv(np.fft.fft(B, axis=0))
    return np.real(np.fft.ifft(H, axis=0))[:n]


def generate_fft(A, Sig, n, rng, Psi=None):
    """Like generate() for any VAR: MA form (impulse_fft) by FFT convolution; start-up discarded."""
    p, m, _ = A.shape
    Psi = impulse_fft(A) if Psi is None else Psi
    burn = len(Psi)
    L = np.linalg.cholesky(Sig)
    e = rng.standard_normal((n + burn, m)) @ L.T
    out = np.zeros((n + burn, m))
    for i in range(m):
        for j in range(m):
            out[:, i] += fftconvolve(e[:, j], Psi[:, i, j])[:n + burn]
    return out[burn:]
