"""Pure algebra of the encoder-transient maps: numpy/torch only, NO project imports (ET-006).

Shared by the simulation study (recon.py, sched_encoder.py, which re-export these names) and the
Telica study (telica/), which runs on telica-real's vendored stack with Telica parameters and so must
not import the simulation vendor.
"""
__project_origin__ = "added"

import numpy as np
import torch
from torch import nn

S_SCALE = 0.40          # HEURISTIC: s = Y / 0.40 keeps the operating range [-0.36, 0.36] inside [-1, 1]
FIT_LO, FIT_HI, FIT_STEP = -0.36, 0.36, 0.005   # D-167: fit over the measured operating range


def recon_map(A, B, C, D, n):
    """(W_y, W_u) in whatever frame (A, B, C, D) are given in. Same algebra as
    linear_encoder_init_aug, with the Markov parameters computed once (identical values)."""
    nx, nu, ny = A.shape[0], B.shape[1], C.shape[0]
    Apow = [np.eye(nx)]
    for _ in range(n):
        Apow.append(Apow[-1] @ A)
    # THEORY: Hoekstra 2026 Eq. 10, T_n: block (n-i, n-j) = C A^(j-i-1) B for j > i, D on the diagonal
    markov = [C @ Apow[m] @ B for m in range(n)]
    Gamma = np.zeros(((n + 1) * ny, (n + 1) * nu))
    for i in range(n + 1):
        for j in range(i, n + 1):
            fi, fj = n - i, n - j
            Gamma[fi * ny:(fi + 1) * ny, fj * nu:(fj + 1) * nu] = markov[j - i - 1] if i != j else D
    # THEORY: Hoekstra 2026 Eq. 10, O_n = [C; CA; ...; CA^n]
    O = np.vstack([C @ Apow[i] for i in range(n + 1)])
    Oinv = np.linalg.pinv(O)
    An = Apow[n]
    # THEORY: Hoekstra 2026 Eq. 13-14, r_n
    gam = np.zeros((nx, (n + 1) * nu))
    for i in range(n):
        gam[:, i * nu:(i + 1) * nu] = Apow[n - i - 1] @ B
    Wy = An @ Oinv
    Wu = -An @ Oinv @ Gamma + gam
    return Wy, Wu


def pure_map(Ad, Bd, Cd, Dd, n, sx, su, sy):
    """The map in the encoder's pure-scaled frame, exactly as normalize_linear_ss_matrices +
    linear_encoder_init_aug form it (float64)."""
    Tx, Tix = np.diag(1 / sx), np.diag(sx)
    return recon_map(Tx @ Ad @ Tix, Tx @ Bd @ np.diag(su), np.diag(1 / sy) @ Cd @ Tix,
                     np.diag(1 / sy) @ Dd @ np.diag(su), n)


def cheb_basis(s, deg):
    """[T_1(s) - T_1(0), ..., T_deg(s) - T_deg(0)], s (N,) -> (N, deg). Works on numpy or torch."""
    lib = torch if isinstance(s, torch.Tensor) else np
    T = [lib.ones_like(s), s]
    for _ in range(2, deg + 1):
        T.append(2 * s * T[-1] - T[-2])
    T0 = [1.0, 0.0]
    for _ in range(2, deg + 1):
        T0.append(2 * 0.0 * T0[-1] - T0[-2])
    cols = [T[k] - T0[k] for k in range(1, deg + 1)]
    if not cols:
        return lib.zeros((len(s), 0), dtype=s.dtype) if lib is np else torch.zeros(len(s), 0, dtype=s.dtype)
    return lib.stack(cols, 1) if lib is torch else np.stack(cols, 1)


def grid_maps(builder, dt, n, sx, su, sy, lo=FIT_LO, hi=FIT_HI, step=FIT_STEP):
    """Exact pure maps on the fit grid. Returns (nodes, W (nodes, nx, M), W0 (nx, M)),
    M = (n+1)(ny+nu), columns [y block | u block]."""
    nodes = np.round(np.arange(lo, hi + step / 2, step), 10)
    W = np.stack([np.concatenate(pure_map(*builder(dt, Y), n, sx, su, sy), axis=1) for Y in nodes])
    W0 = np.concatenate(pure_map(*builder(dt, 0.0), n, sx, su, sy), axis=1)
    return nodes, W, W0


def fit_schedule(grid, deg):
    """Least-squares C_k with W(0) pinned, from `grid_maps`. Returns (C (deg, nx, M), W0, residual)."""
    nodes, W, W0 = grid
    D = (W - W0[None]).reshape(len(nodes), -1)
    Phi = cheb_basis(nodes / S_SCALE, deg)
    if deg > 0:
        C, *_ = np.linalg.lstsq(Phi, D, rcond=None)
        R = D - Phi @ C
    else:
        C, R = np.zeros((0, D.shape[1])), D
    nrmW = np.linalg.norm(W.reshape(len(nodes), -1), axis=1)
    res = dict(max_rel_to_W=float((np.linalg.norm(R, axis=1) / nrmW).max()),
               rel_to_correction=float(np.linalg.norm(R) / max(np.linalg.norm(D), 1e-300)))
    return C.reshape(deg, *W0.shape), W0, res


def numpy_sched(C, W0, yw, uw, sx, su, sy):
    """The scheduled map's linear part in numpy: physical windows -> physical state."""
    N = len(yw)
    wp = np.concatenate([(yw / sy).reshape(N, -1), (uw / su).reshape(N, -1)], 1)
    x = wp @ W0.T
    if len(C):
        phi = cheb_basis(yw[:, -1, 2] / S_SCALE, len(C))
        x = x + np.einsum('bk,kxm,bm->bx', phi, C, wp)
    return x * sx


class ScheduledReconEncoder(nn.Module):
    """x(k) = parent(u, y) + sum_{k=1..deg} (T_k(s) - T_k(0)) (C^y_k y_pure + C^u_k u_pure), s = Y/S.

    `parent` is a `linear_encoder_init_aug`; with deg = 0 the forward returns parent(u, y) untouched
    (bitwise collapse). Y = measured stage Y at the window's last sample from the encoder's yhist."""

    def __init__(self, parent, C, n, ny, nu, y0, ystd, y_index=2):
        super().__init__()
        self.parent = parent
        self.deg = int(C.shape[0])
        self.n, self.ny, self.nu, self.y_index = n, ny, nu, y_index
        dt = next(parent.parameters()).dtype
        if self.deg:
            My = (n + 1) * ny
            self.Cy = nn.Parameter(torch.as_tensor(C[:, :, :My], dtype=dt))
            self.Cu = nn.Parameter(torch.as_tensor(C[:, :, My:], dtype=dt))
        self.register_buffer('y0Y', torch.as_tensor(float(np.ravel(y0)[y_index]), dtype=dt))
        self.register_buffer('ystdY', torch.as_tensor(float(np.ravel(ystd)[y_index]), dtype=dt))

    def forward(self, uhist, yhist):
        x = self.parent(uhist, yhist)
        if self.deg == 0:
            return x                                              # collapse: the parent, bitwise
        p = self.parent
        B = yhist.shape[0]
        y_pure = yhist.reshape(B, -1) + p.y_off.view(1, -1)       # same pure frame as the parent
        u_pure = uhist.reshape(B, -1) + p.u_off.view(1, -1)
        Y = yhist[:, -1, self.y_index] * self.ystdY + self.y0Y    # measured stage Y at k
        phi = cheb_basis(Y / S_SCALE, self.deg)                   # (B, deg)
        corr = torch.einsum('bk,kxm,bm->bx', phi, self.Cy, y_pure) \
            + torch.einsum('bk,kxm,bm->bx', phi, self.Cu, u_pure)
        r = corr.shape[1]                    # baseline source: 6 scheduled rows, W^a rows untouched
        return torch.cat([x[:, :r] + corr, x[:, r:]], dim=1)


def grid_apply(grid, yw, uw, Y, sx, su, sy):
    """Exact frozen-Y maps from `grid_maps`, linearly interpolated at a per-window Y (pure frame
    inside, physical windows in, physical state out). Y outside the grid is clipped."""
    nodes, W, _ = grid
    step = nodes[1] - nodes[0]
    Yc = np.clip(np.asarray(Y, float), nodes[0], nodes[-1] - 1e-12)
    pos = (Yc - nodes[0]) / step
    i = np.floor(pos).astype(int)
    t = (pos - i)[:, None]
    N = len(yw)
    wp = np.concatenate([(yw / sy).reshape(N, -1), (uw / su).reshape(N, -1)], 1)
    x = np.zeros((N, W.shape[1]))
    for node in np.unique(i):
        m = i == node
        x[m] = (1 - t[m]) * (wp[m] @ W[node].T) + t[m] * (wp[m] @ W[node + 1].T)
    return x * sx
