"""Shared spectral tools for closed-loop-noise (CN-004): records, windows, Welch matrices, bands, CIs.

Conventions: signals (N, 3) in stage axes [X1, X2, Y]; spectral matrices P[f, i, j] = E[X_i X_j^*]
(one-sided density, m^2/Hz for positions), so the shaping of a source v by a 3x3 transfer H is
P_out = H P_v H^H at every frequency.
"""
import numpy as np
from scipy import stats
from scipy.signal import get_window

import cn_env
import data_index
from real_data_verification.telica_loader import load_telica_log_cl

FS = cn_env.FS
NPER = 2048                       # CN-004
NOV = 1024
GUARD = 200                       # CN-003 / CN-004: window ends GUARD samples before j0
HANN_RHO = 0.167                  # THEORY: Harris 1978 Table 1, Hann 50 % overlap correlation
COLORS = ('#2a78d6', '#eb6834', '#1baf7a')     # X1, X2, Y (dataviz reference palette, slots 1-3)
INK = '#3d3d3a'


def first_change(r):
    """j0: first index with r[j0 + 1] != r[j0] on any axis (09-22 probe, CN-003)."""
    mv = np.abs(np.diff(r, axis=0)).sum(1) > 0
    return int(np.argmax(mv)) if mv.any() else len(r) - 1


def all_records():
    """[(split, op, it, path)] for every log of the split (CN-004), incl. the iter6_1 redo."""
    recs = data_index.records(splits=('train', 'validation', 'test'),
                              iters=tuple(data_index.ITERS_COMMON), optional=True)
    p = data_index.path('train', 'xpos_-60_ypos120', 'iter6_1')
    import os
    if os.path.isfile(p):
        recs.append(('train', 'xpos_-60_ypos120', 'iter6_1', p))
    return recs


def load(path):
    """Whole record as numpy float64: r, q1, e [m]; u_ff [N]; i_fb [A]; j0."""
    d = load_telica_log_cl(path, pre_motion_ms=None, engine='c')
    out = {k: d[k].numpy() for k in ('r', 'q1', 'e', 'u_ff', 'i_fb')}
    out['j0'] = first_change(out['r'])
    return out


def welch_sum(x, nper=NPER, nov=NOV, fs=FS):
    """Welch cross-spectral SUM over segments. x (N, m) -> (f, Psum (F, m, m) complex, K).

    Psum / K is the usual Welch estimate (Hann periodic window, constant detrend, one-sided
    density). Returning the sum lets records be pooled with every segment counting once.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    w = get_window('hann', nper)
    step = nper - nov
    K = (len(x) - nper) // step + 1
    f = np.fft.rfftfreq(nper, 1 / fs)
    if K < 1:
        return f, np.zeros((len(f), x.shape[1], x.shape[1]), complex), 0
    idx = np.arange(nper)[None, :] + step * np.arange(K)[:, None]
    seg = x[idx]                                          # (K, nper, m)
    seg = seg - seg.mean(1, keepdims=True)
    X = np.fft.rfft(seg * w[None, :, None], axis=1)       # (K, F, m)
    P = np.einsum('kfi,kfj->fij', X, X.conj()) / (fs * np.sum(w ** 2))
    P[1:] *= 2.0
    if nper % 2 == 0:
        P[-1] /= 2.0
    return f, P, K


def nu_welch(K, rho=HANN_RHO):
    """Equivalent chi-square dof of K overlapped Hann segments (THEORY: Welch 1967; Harris 1978)."""
    return 2.0 * K / (1.0 + 2.0 * rho ** 2)


def chi2_ci(P, nu, alpha=0.05):
    lo = nu * P / stats.chi2.ppf(1 - alpha / 2, nu)
    hi = nu * P / stats.chi2.ppf(alpha / 2, nu)
    return lo, hi


def bands(f, lo=10.0, hi=10000.0, n_edges=61, min_bins=2):
    """Band index per bin (-1 = outside) and band centre/edge arrays (CN-004, HEURISTIC)."""
    edges = np.geomspace(lo, hi, n_edges)
    raw = np.digitize(f, edges) - 1
    raw[(f < lo) | (f >= hi)] = -1
    groups, cur = [], []
    for b in range(n_edges - 1):
        cur += list(np.where(raw == b)[0])
        if len(cur) >= min_bins:
            groups.append(cur); cur = []
    if cur and groups:
        groups[-1] += cur
    band_of = np.full(len(f), -1)
    for i, g in enumerate(groups):
        band_of[g] = i
    fc = np.array([np.sqrt(f[g[0]] * f[g[-1]]) for g in groups])
    fe = np.array([[f[g[0]], f[g[-1]]] for g in groups])
    return band_of, fc, fe


def band_mean(P, band_of, nb=None):
    """Average of P over the bins of each band along axis 0."""
    nb = band_of.max() + 1 if nb is None else nb
    out = np.zeros((nb,) + P.shape[1:], dtype=P.dtype)
    for b in range(nb):
        out[b] = P[band_of == b].mean(0)
    return out


def autos(P):
    """(F, 3, 3) -> (F, 3) real auto-spectra."""
    return np.real(np.einsum('fii->fi', P))


def shape(H, P):
    """H (F, m, m) applied to spectral matrix P (F, m, m): H P H^H."""
    return np.einsum('fij,fjk,flk->fil', H, P, H.conj())
