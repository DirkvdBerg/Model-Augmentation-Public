"""Orthogonal projection-based regularization for model augmentation.

Method: Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth (L4DC 2025), Section 4
(Taylor/extended-regressor variant), Eqs. 13-14 and 19, adapted to the
routed-row LFR setting. Design + approval trail:
docs/orthogonal-projection-plan.md, Sect. 7, "Step 7 DELIVERABLE" (D7.1-D7.9).
Formula validation: Stage A ladder, scripts/gantry/orth-projection/ steps 0-6
(results in simulations/gantry_subnet/diagnostics/orth_projection/).

Contents:
  OrthProjectionPenalty      the penalty term V_orth = beta * ||Q^T f_ANN||^2
  SSE_Interconnect_OrthLoss  ParamLoss subclass adding the penalty when
                             configured; exact no-op otherwise (D7.2)
"""
__project_origin__ = "added"

import torch
from torch import nn, Tensor

from model_augmentation.fit_systems.blocks import Static_ANN_Block
from model_augmentation.fit_systems.interconnect import SSE_Interconnect_ParamLoss


class OrthProjectionPenalty(nn.Module):
    """V_orth = beta * || Q^T stack(f_ANN(Z_pts)[:, route_cols]) ||^2 (D7.1).

    Parameters
    ----------
    Q : array (N_pts * n_r, k)
        Rank-truncated basis of the extended regressor [Phi_theta | Gamma]
        at theta_bar, rows restricted to the routed physical state rows,
        stacked sample-major (plan Sect. 2.2; Stage A steps 2-4).
    Z_pts : array (N_pts, nz, 1)
        Fixed normalized ANN inputs [x_phys, x_aug=0, u] on the
        theta_bar / zero-ANN manifold (D7.4). nz = nxd + nu.
    route_cols : list[int]
        ANN output columns that map to state rows < nx_phys, in
        ann_route_ix order; n_r = len(route_cols). Output columns mapping to
        augmented rows carry zero baseline signature and are unpenalized by
        construction (plan Sect. 2.3).
    beta : float
        Penalty weight, added to the loss per batch, undivided (D7.6): any
        constant scale is absorbed by the swept beta.
    dtype : torch dtype
        Pipeline dtype for the buffers (D7.7: Q computed in f64, cast here).
    """

    def __init__(self, Q, Z_pts, route_cols, beta, dtype=torch.float32):
        super().__init__()
        self.register_buffer("Q", torch.as_tensor(Q, dtype=dtype))
        self.register_buffer("Z_pts", torch.as_tensor(Z_pts, dtype=dtype))
        self.route_cols = list(route_cols)
        self.beta = float(beta)

    def penalty_of_field(self, f_stacked: Tensor) -> Tensor:
        """beta * ||Q^T f||^2 for an already-stacked output field.

        Computed without materializing the projector (Gyorok Eq. 14).
        Exposed separately so diagnostics can drive it with synthetic fields
        (Step 8 parity check 1).
        """
        return self.beta * torch.linalg.vector_norm(self.Q.T @ f_stacked) ** 2

    def forward(self, ann_block: Static_ANN_Block) -> Tensor:
        w = ann_block(self.Z_pts)                        # (N_pts, nw, 1)
        f = w[:, self.route_cols, 0].reshape(-1)         # sample-major stack
        return self.penalty_of_field(f)


class SSE_Interconnect_OrthLoss(SSE_Interconnect_ParamLoss):
    """ParamLoss + optional orthogonal-projection penalty (D7.1/D7.2).

    `orth_penalty` is None by default: loss() returns the parent value
    untouched (bit-identical no-op, Step 8 parity check 2). The pipeline
    attaches an OrthProjectionPenalty AFTER construction when
    cfg.orth_beta > 0; the penalty is NOT part of hfn/encoder, so it never
    enters deepSI checkpoints and pre-hook checkpoints resume unchanged
    (D7.8).

    The penalty is a function of the ANN block only (Q and Z_pts are fixed
    buffers): its gradient into the physical parameters and the encoder is
    exactly zero (Step 8 parity check 3).
    """

    orth_penalty = None   # class default; instance attribute set by the pipeline

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        L = super().loss(uhist, yhist, ufuture, yfuture, **Loss_kwargs)
        if self.orth_penalty is not None and self.orth_penalty.beta != 0.0:
            ann = next(m for m in self.hfn.connected_blocks
                       if isinstance(m, Static_ANN_Block))
            L = L + self.orth_penalty(ann)
        return L
