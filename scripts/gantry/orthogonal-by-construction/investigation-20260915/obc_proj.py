"""Shared prototype pieces for the orthogonal-by-construction investigation.

__project_origin__ = "added"

The learned block, the stacked baseline regressor at a declared expansion point, the
declared pseudo-inverse convention, and the projection itself. Imported by the stage
scripts so the same objects are used everywhere; nothing here runs on import.
"""

__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import obc_common as gc                                            # noqa: E402

NP_, NA, NU = 6, 2, 3
RCOND = 1e-10


class Tee:
    """stdout mirror to a log file."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s); self.f.write(s)

    def flush(self):
        sys.__stdout__.flush(); self.f.flush()


class ANN(torch.nn.Module):
    """Small tanh MLP reading (x_p, x_a, u), writing n_p + n_a rows."""

    def __init__(self, nh=16, seed=0, w_scale=0.5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.l1 = torch.nn.Linear(NP_ + NA + NU, nh)
        self.l2 = torch.nn.Linear(nh, nh)
        self.l3 = torch.nn.Linear(nh, NP_ + NA)
        for m in (self.l1, self.l2, self.l3):
            torch.nn.init.normal_(m.weight, 0.0, w_scale, generator=g)
            torch.nn.init.normal_(m.bias, 0.0, 0.1, generator=g)
        self.register_buffer("sx", torch.tensor([0.3, 0.02, 0.3, 0.5, 0.05, 0.5, 1.0, 1.0]))
        self.register_buffer("su", torch.tensor([100.0, 100.0, 100.0]))
        # Write scale: the learned write is a per-step state increment. Position rows are
        # given a small scale and velocity rows a larger one, matching the magnitudes of
        # the nominal increment measured in stage 1 (max |x_next - x| = 1.74e-3).
        self.register_buffer("sw", torch.tensor([1e-6, 1e-6, 1e-6,
                                                 1e-3, 1e-3, 1e-3, 1.0, 1.0]))

    def forward(self, xp, xa, u):
        z = torch.cat([xp / self.sx[:6], xa / self.sx[6:], u / self.su], dim=-1)
        h = torch.tanh(self.l1(z))
        h = torch.tanh(self.l2(h))
        return self.l3(h) * self.sw


def flat_params(net):
    return torch.cat([p.reshape(-1) for p in net.parameters()])


def g_write(net, params_flat, xp, xa, u):
    shapes = [p.shape for p in net.parameters()]
    names = [n for n, _ in net.named_parameters()]
    i, d = 0, {}
    for n, s in zip(names, shapes):
        k = int(np.prod(s))
        d[n] = params_flat[i:i + k].reshape(s)
        i += k
    buf = {n: b for n, b in net.named_buffers()}
    return torch.func.functional_call(net, {**d, **buf}, (xp, xa, u))


def phi_stack(theta_bar, Lb, X, U, coord="log", with_offset=True, Y_frozen=None):
    """Stacked [Phi_p | c] over points, shape (n_p*M, n_theta(+1))."""
    def f(th_free):
        th = (theta_bar * torch.exp(th_free) if coord == "log"
              else theta_bar * th_free if coord == "normalized" else th_free)
        return gc.rk4_step(th, Lb, X, U, Y_frozen=Y_frozen).reshape(-1)

    base = (torch.zeros_like(theta_bar) if coord == "log"
            else torch.ones_like(theta_bar) if coord == "normalized" else theta_bar)
    # Forward mode: 14 tangents, against n_p*M cotangents for reverse mode. At M = 1024
    # that is 14 passes instead of 6144, which is the difference between seconds and
    # tens of minutes.
    Phi = torch.func.jacfwd(f)(base)
    if not with_offset:
        return Phi
    f0 = f(base).detach()
    c = (f0 - Phi @ theta_bar) if coord == "physical" else f0
    return torch.cat([Phi, c.unsqueeze(-1)], dim=-1)


def pinv_report(Phi, rcond=RCOND, label=None):
    U_, S_, Vh = torch.linalg.svd(Phi, full_matrices=False)
    keep = S_ > rcond * S_[0]
    r = int(keep.sum())
    if label is not None:
        disc = S_[~keep]
        tail = (f"largest discarded / sigma0 = {(disc.max() / S_[0]).item():.2e}"
                if len(disc) else "none discarded")
        print(f"    {label}: shape {tuple(Phi.shape)} rank {r}/{len(S_)} "
              f"at rcond {rcond:.0e}; {tail}")
    Sinv = torch.where(keep, 1.0 / S_, torch.zeros_like(S_))
    K = Vh.T @ torch.diag(Sinv) @ U_.T
    return K, r, S_


def proj_basis(Phi, rcond=RCOND):
    """Orthonormal basis of range(Phi), rank truncated."""
    U_, S_, _ = torch.linalg.svd(Phi, full_matrices=False)
    r = int((S_ > rcond * S_[0]).sum())
    return U_[:, :r], r


def rho(Phi, G, rcond=RCOND):
    """Fraction of ||G|| lying in range(Phi). Scale free, in [0, 1]."""
    Q, _ = proj_basis(Phi, rcond)
    return (torch.linalg.norm(Q @ (Q.T @ G)) / torch.linalg.norm(G)).item()


def design_latent(n, seed, scale=1.0):
    return torch.tensor(np.random.default_rng(seed).uniform(-1, 1, size=(n, NA)) * scale)
