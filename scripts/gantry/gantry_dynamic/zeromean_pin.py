"""Zero-mean pin for the K=0 (X/Y) ANN output rows -- the INTERVENTIONAL demonstrator (G9).

Soft penalty added to the training loss via the existing `fit_sys.orth_penalty` hook
(SSE_Interconnect_OrthLoss.loss adds any attached penalty with .beta != 0; no framework edits):

    V_pin = beta * sum_{r in K=0 routed rows} ( mean_j  w_r(Z_j) )^2

evaluated on a FIXED set of normalized ANN input points Z_pts from the training data manifold
(same construction as orth_penalty's Z_pts: data-derived logical states + x_aug=0 + u; the hook
calls penalty(ann) with no batch context, so the penalty carries its own points).

WHAT IT IS / IS NOT: it penalizes only the ensemble-MEAN of the output (the drift-driving
constant); the ANN remains fully expressive for any zero-mean force. Valid as a DEMONSTRATION
on the frictionless sim (the true residual is genuinely zero-mean, d12: the mean direction is
loss-neutral); NOT the thesis deliverable -- on real data friction carries a legitimate nonzero
mean force, so the deliverable is the frequency-selective direction pin (d16 / concept §7).

Must live in an importable module (not a run script): checkpoint_save_system pickles the whole
fit_sys.__dict__ including the attached penalty.
"""
__project_origin__ = "added"

import numpy as np
import torch
from torch import nn

# K=0 physical state rows (X, Y positions + velocities); pinned cols = their positions
# in ann_route_ix order.
K0_ROWS = (0, 2, 3, 5)

# HEURISTIC: beta such that the pin term at the OBSERVED unpinned DC (~1.4e-6 normalized,
# gantry_drift_last dY row) is ~10% of the typical training loss (~1.4e-6):
# beta * (1.4e-6)^2 ~ 1.4e-7 -> beta ~ 7e4. Because the mean direction is loss-neutral
# (d12), the result should be insensitive to beta over several decades -- part of what the
# run verifies.
BETA_DEFAULT = 7e4


class ZeroMeanPin(nn.Module):
    """V_pin = beta * sum_r mean_j(w_r(Z_j))^2 over the K=0 routed output columns."""

    def __init__(self, Z_pts, route_cols, beta=BETA_DEFAULT, dtype=torch.float32):
        super().__init__()
        self.register_buffer("Z_pts", torch.as_tensor(Z_pts, dtype=dtype))
        self.route_cols = list(route_cols)
        self.beta = float(beta)
        # Compatibility shim for the [joint-probe] meter (reads pen.Q): zero basis ->
        # orth-frac prints 0; the pin's own value is printed via the [orth-probe] line.
        n_stack = self.Z_pts.shape[0] * len(self.route_cols)
        self.register_buffer("Q", torch.zeros((n_stack, 1), dtype=dtype))

    def forward(self, ann_block):
        w = ann_block(self.Z_pts)                       # (N_pts, nw, 1)
        means = w[:, self.route_cols, 0].mean(dim=0)    # (n_pinned,)
        return self.beta * (means ** 2).sum()

    @torch.no_grad()
    def row_means(self, ann_block):
        w = ann_block(self.Z_pts)
        return w[:, self.route_cols, 0].mean(dim=0).cpu().numpy()


def build_zeromean_pin(cfg, data, norm, beta=BETA_DEFAULT, stride=100, verbose=True):
    """Z_pts from the training records (data-derived states, x_aug=0, normalized u)."""
    from .orth_penalty import _x_logical_from_data   # same state construction (D-111)
    xs, us = [], []
    for sd in data.train_list:
        xl = _x_logical_from_data(sd)                                  # (N,6) physical
        xn = (xl - norm.x_mean.flatten()) / norm.std_x.flatten()       # normalized
        un = (np.asarray(sd.u) - norm.u_mean.flatten()) / norm.std_u.flatten()
        K0 = cfg.na_nb
        xs.append(xn[K0::stride]); us.append(un[K0::stride])
    X = np.concatenate(xs); U = np.concatenate(us)
    Z = np.concatenate([X, np.zeros((len(X), cfg.nx_ann)), U], axis=1)[:, :, None]
    route_cols = [i for i, r in enumerate(cfg.ann_route_ix) if r in K0_ROWS]
    pin = ZeroMeanPin(Z, route_cols, beta=beta, dtype=cfg.dtype_pt)
    if verbose:
        print(f'[zeromean-pin] {Z.shape[0]} points (stride {stride}), pinned output cols '
              f'{route_cols} (state rows {[cfg.ann_route_ix[c] for c in route_cols]}), '
              f'beta={beta:.1e} (HEURISTIC, see module docstring)')
    return pin
