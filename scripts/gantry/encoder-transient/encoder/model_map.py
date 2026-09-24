"""The model-consistent scheduled map, built from ANY model the encoder feeds (ET-008).

G2 built the planted map from the planted model's analytic linearisation. In training the model is
baseline + ANN and changes every update, so the map has to come from the model itself:

    for each Y on the fit grid:
        x_op = rest at Y (physical [0, 0, Y, 0, 0, 0], latent rows 0), u_op = zero physical force
        A = d x_next / d x,  B = d x_next / d u,  C = d y / d x     (autograd, the model's own step)
        W(Y) = recon_map(A, B, C, 0, n)                              (Hoekstra 2026 Eq. 16-17)
    Chebyshev fit in Y with W(0) pinned (core.fit_schedule), written into a ScheduledReconEncoder:
    parent.W^b = W0[:6], parent.W^a = W0[6:], C_k for ALL rows.

The Jacobians are taken in the model's normalised frame, which differs from the encoder's pure-scaled
frame only by constant offsets, so A, B, C are the same in both (the offsets cancel in derivatives).
Rows the model cannot see (a zero-initialised ANN leaves the latent states unobservable) come out as
zero rows of O_n^+, i.e. the minimum-norm, model-consistent W^a = 0.

Only torch/numpy and core.py: usable with the simulation vendor and with the Telica vendor.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core                                                      # noqa: E402


def op_point(Y, norm, nx, dtype):
    """Rest at Y in the model's normalised frame, and zero physical force."""
    xm = np.asarray(norm.x_mean, float).ravel()
    sx = np.asarray(norm.std_x, float).ravel()
    x = np.zeros(nx)
    phys = np.array([0.0, 0.0, Y, 0.0, 0.0, 0.0])
    x[:6] = (phys - xm) / sx
    u = -np.asarray(norm.u_mean, float).ravel() / np.asarray(norm.std_u, float).ravel()
    return torch.as_tensor(x, dtype=dtype), torch.as_tensor(u, dtype=dtype)


def linearise(hfn, Y, norm, nx, dtype=torch.float64):
    """(A, B, C, offset) of the model step at rest at Y. `offset` = x_next(x_op, u_op) - x_op,
    the affine term a linear map cannot carry (0 for the baseline at rest; reported, not used)."""
    x0, u0 = op_point(Y, norm, nx, dtype)

    def step(x, u):
        return hfn(x[None], u[None])[1][0]

    def out(x):
        return hfn.output_only(x[None])[0]

    with torch.enable_grad():
        A, B = torch.autograd.functional.jacobian(step, (x0, u0))
        C = torch.autograd.functional.jacobian(out, x0)
    with torch.no_grad():
        off = step(x0, u0) - x0
    return A.double().numpy(), B.double().numpy(), C.double().numpy(), off.double().numpy()


def model_grid(hfn, norm, nx, n, lo=core.FIT_LO, hi=core.FIT_HI, step=core.FIT_STEP, dtype=torch.float64):
    """(nodes, W (nodes, nx, M), W0 (nx, M)) in the pure frame, and the max affine offset seen."""
    nodes = np.round(np.arange(lo, hi + step / 2, step), 10)
    Ws, offs = [], []
    ny, nu = 3, 3

    def one(Y):
        A, B, C, off = linearise(hfn, Y, norm, nx, dtype)
        Wy, Wu = core.recon_map(A, B, C, np.zeros((ny, nu)), n)
        offs.append(float(np.abs(off).max()))
        return np.concatenate([Wy, Wu], 1)
    for Y in nodes:
        Ws.append(one(Y))
    W0 = one(0.0)
    return (nodes, np.stack(Ws), W0), max(offs)


@torch.no_grad()
def write_into(enc, grid, deg):
    """Overwrite a ScheduledReconEncoder's linear part with the fitted model-consistent map.
    enc.parent must have W^b (6 rows) + W^a (nx - 6 rows), and enc.Cy/Cu shaped (deg, nx, .)."""
    C, W0, res = core.fit_schedule(grid, deg)
    p = enc.parent
    My = (enc.n + 1) * enc.ny
    dt = p.Wb_psi_y.dtype
    p.Wb_psi_y.copy_(torch.as_tensor(W0[:6, :My], dtype=dt))
    p.Wb_psi_u.copy_(torch.as_tensor(W0[:6, My:], dtype=dt))
    p.Wa_psi_y.copy_(torch.as_tensor(W0[6:, :My], dtype=dt))
    p.Wa_psi_u.copy_(torch.as_tensor(W0[6:, My:], dtype=dt))
    if enc.deg:
        enc.Cy.copy_(torch.as_tensor(C[:, :, :My], dtype=dt))
        enc.Cu.copy_(torch.as_tensor(C[:, :, My:], dtype=dt))
    return res


def consistent_encoder(parent, hfn, norm, nx, n, deg, y0, ystd, dtype=torch.float64):
    """A ScheduledReconEncoder on `parent` (a linear_encoder_init_aug with nx rows), initialised
    from the model `hfn`. Returns (encoder, fit residual, max affine offset)."""
    grid, off = model_grid(hfn, norm, nx, n, dtype=dtype)
    M = grid[2].shape[1]
    enc = core.ScheduledReconEncoder(parent, np.zeros((deg, nx, M)), n, 3, 3, y0, ystd).to(dtype)
    res = write_into(enc, grid, deg)
    return enc, res, off


def reset_adam_state(optimizer, params):
    """Stale moments after an overwrite: drop Adam's state for these parameters only."""
    ids = {id(p) for p in params}
    for p in list(optimizer.state.keys()):
        if id(p) in ids:
            optimizer.state[p] = {}


class RefreshingLoss:
    """Picklable replacement for `fit_sys.loss` (deepSI checkpoints pickle the system's __dict__,
    so a closure cannot be used): before the loss of update k, rebuild the encoder's linear part
    from the CURRENT model whenever k crosses a multiple of `every` (k = 0 included)."""

    def __init__(self, fs, norm, nx, n, every):
        self.fs, self.norm, self.nx, self.n, self.every = fs, norm, nx, n, max(1, int(every))
        self.done, self.log = set(), []

    def refresh(self, k):
        import copy
        import time
        slot = k // self.every
        if slot in self.done:
            return
        t = time.time()
        hfn64 = copy.deepcopy(self.fs.hfn).to('cpu', torch.float64)
        grid, off = model_grid(hfn64, self.norm, self.nx, self.n)
        res = write_into(self.fs.encoder, grid, self.fs.encoder.deg)
        reset_adam_state(self.fs.optimizer, list(self.fs.encoder.parameters()))
        self.done.add(slot)
        self.log.append(dict(update=k, fit_resid=res['max_rel_to_W'], offset=off, seconds=time.time() - t))
        print(f'[refresh] update {k}: model-Jacobian map rebuilt (resid {res["max_rel_to_W"]:.1e}, '
              f'max affine offset {off:.1e}, {time.time() - t:.1f} s)', flush=True)

    def __call__(self, *a, **kw):
        self.refresh(int(getattr(self.fs, 'batch_counter', 0)))
        return type(self.fs).loss(self.fs, *a, **kw)


def freeze_linear(enc):
    """ET-013: requires_grad = False on the encoder's linear map (W^b, W^a, the Y-schedule C_k);
    the correction net stays trainable. Returns (frozen, trainable) parameter counts."""
    frozen = trainable = 0
    for name, p in enc.named_parameters():
        if any(k in name for k in ('Wb_psi', 'Wa_psi', 'Cy', 'Cu')):
            p.requires_grad_(False)
            frozen += p.numel()
        elif p.requires_grad:
            trainable += p.numel()
    return frozen, trainable
