"""The Telica BHL feedback controller, from the zpk file to a well-conditioned state space (TR-008).

    K_j(z): error e = r - y [m] -> current [A], j in (LX1, LX2, LY), Ts = 5e-5 s (20 kHz only).

Built from `telica_zpk_bhl.mat` (exported from kamtin-data/dFeedbackControllersTelica.mat by
export_zpk_bhl.m). The near-unit "integrator" poles are snapped to exactly z = 1. Each axis is a
cascade of second-order sections (scipy zpk2sos pairing), each section in direct-form-II-
transposed state space; the cascade is one (A, B, C, D) per axis, block-diagonal over the axes.
"""
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import zpk2sos, sosfilt, sosfilt_zi, sos2zpk

from params import telica_params as tp

HERE = os.path.dirname(os.path.abspath(__file__))
ZPK_FILE = os.path.join(HERE, 'telica_zpk_bhl.mat')
AXES = ('LX1', 'LX2', 'LY')
SNAP_TOL = 1e-6          # |p - 1| below this is an integrator by design (TR-008)


def load_zpk(snap=True):
    """{axis: (z, p, k)} and Ts. `snap` puts the near-unit real poles exactly at z = 1."""
    m = loadmat(ZPK_FILE, squeeze_me=True)
    Ts = float(m['Ts'])
    out, snapped = {}, {}
    for i, ax in enumerate(AXES):
        z = np.atleast_1d(np.asarray(m['z'][i], dtype=complex))
        p = np.atleast_1d(np.asarray(m['p'][i], dtype=complex))
        k = float(np.atleast_1d(m['k'])[i])
        if snap:
            near = np.abs(p - 1.0) < SNAP_TOL
            snapped[ax] = [float(abs(q)) for q in p[near]]
            p = np.where(near, 1.0 + 0j, p)
        out[ax] = (z, p, k)
    if abs(Ts - tp.TS) > 1e-15:
        raise ValueError(f'controller Ts {Ts} != params Ts {tp.TS}')
    return out, Ts, snapped


def sos_bank(snap=True):
    """{axis: sos (n_sec, 6)} in scipy convention."""
    zpk, _, _ = load_zpk(snap)
    out = {}
    for ax, (z, p, k) in zpk.items():
        out[ax] = zpk2sos(z, p, k, pairing='nearest')
    return out


def _section_ss(sec):
    """DF2T state space of one biquad (b0 b1 b2 1 a1 a2)."""
    b0, b1, b2, a0, a1, a2 = sec / sec[3]
    A = np.array([[-a1, 1.0], [-a2, 0.0]])
    B = np.array([b1 - a1 * b0, b2 - a2 * b0])
    C = np.array([1.0, 0.0])
    D = b0
    return A, B, C, D


def axis_ss(sos):
    """Cascade of sections -> one SISO (A, B, C, D). Section i feeds section i+1."""
    A = np.zeros((0, 0)); B = np.zeros(0); C = np.zeros(0); D = 1.0
    for sec in sos:
        As, Bs, Cs, Ds = _section_ss(sec)
        n = A.shape[0]
        # new input = previous output y = C x + D u ; x_s' = As x_s + Bs y
        A2 = np.zeros((n + 2, n + 2))
        A2[:n, :n] = A
        A2[n:, :n] = np.outer(Bs, C)
        A2[n:, n:] = As
        B2 = np.concatenate([B, Bs * D])
        C2 = np.concatenate([Ds * C, Cs])
        D2 = Ds * D
        A, B, C, D = A2, B2, C2, D2
    return A, B, C, D


def bank_ss(snap=True, a_to_n=None):
    """3-in 3-out block-diagonal (A, B, C, D), error [m] -> current [A] (or force [N] if
    `a_to_n` per-axis gains are given)."""
    sos = sos_bank(snap)
    blocks = [axis_ss(sos[ax]) for ax in AXES]
    n = sum(b[0].shape[0] for b in blocks)
    A = np.zeros((n, n)); B = np.zeros((n, 3)); C = np.zeros((3, n)); D = np.zeros((3, 3))
    i = 0
    for j, (Aj, Bj, Cj, Dj) in enumerate(blocks):
        m = Aj.shape[0]
        A[i:i + m, i:i + m] = Aj
        B[i:i + m, j] = Bj
        C[j, i:i + m] = Cj
        D[j, j] = Dj
        i += m
    if a_to_n is not None:
        g = np.asarray(a_to_n, float)
        C = g[:, None] * C
        D = g[:, None] * D
    return A, B, C, D


def replay(e, snap=True):
    """Zero-state replay, float64: e (T,3) [m] -> current (T,3) [A]."""
    sos = sos_bank(snap)
    return np.stack([sosfilt(sos[ax], e[:, j]) for j, ax in enumerate(AXES)], 1)


def zir_basis(T, snap=True):
    """Zero-input response basis per axis: {axis: Phi (T, n_states)}, Phi[k, i] = C A^k e_i."""
    A, B, C, D = bank_ss(snap)
    out = {}
    i0 = 0
    for j, ax in enumerate(AXES):
        m = 2 * len(sos_bank(snap)[ax])
        Aj = A[i0:i0 + m, i0:i0 + m]; Cj = C[j, i0:i0 + m]
        Phi = np.empty((T, m))
        row = Cj.copy()
        for k in range(T):
            Phi[k] = row
            row = row @ Aj
        out[ax] = Phi
        i0 += m
    return out


def impulse_energy(n=200000, snap=True):
    """sum_k h_k^2 per axis, for the quantization-noise variance through K (TR-009)."""
    sos = sos_bank(snap)
    out = {}
    x = np.zeros(n); x[0] = 1.0
    for ax in AXES:
        h = sosfilt(sos[ax], x)
        out[ax] = float(np.sum(h ** 2))
    return out
