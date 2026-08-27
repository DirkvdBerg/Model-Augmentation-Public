"""BoundedIntegral_ANN_Block -- the integrator-bounding ANN block for D-C.

__project_origin__ = "added"

Construction (drift-diagnosis-status.md §5b, "bounded-integral channel"):
    g_k = psi(z_k) - psi(z_{k-1}),   psi = tanh(net(z))   (bounded in (-1,1))
The running sum telescopes: sum_{j<=k} g_j = psi(z_k) - psi(z_0), bounded for all k.
On a free integrator this makes the ANN's accumulated contribution bounded -> NO drift, by
construction, while the INSTANTANEOUS output g_k is unconstrained (can be nonzero at rest).

This is the STATEFUL version: the block carries psi_{k-1} across timesteps within one rollout.
`reset()` clears it at the start of every rollout (each training window / sim starts fresh with
x = encoder(...)); the block also auto-resets if the batch size changes or on the first call, in
which case g_0 = 0 and psi_{-1} := psi(z_0).

NOTE: psi_{k-1} is kept as a LIVE tensor (not a buffer) so BPTT connects the difference; it is
detached only at reset. The block is drop-in for Static_ANN_Block (same net, same nz/nw), so the
interconnect wiring is unchanged -- only the forward output is the bounded difference.
"""
__project_origin__ = "added"

import torch
from torch import Tensor

from model_augmentation.fit_systems.blocks import Static_ANN_Block


class BoundedIntegral_ANN_Block(Static_ANN_Block):
    """Static_ANN_Block whose output is the bounded first-difference g_k = psi_k - psi_{k-1}.

    Parameters are identical to Static_ANN_Block (nz, nw, net, n_nodes_per_layer, n_hidden_layers,
    activation). `psi_scale` scales the bounded map psi = psi_scale * tanh(net) so the correction can
    reach the needed magnitude while staying bounded (telescoping stays bounded for any finite scale).
    """

    def __init__(self, *args, psi_scale: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.psi_scale = float(psi_scale)
        self._psi_prev = None      # live tensor (batch, nw, 1) or None; NOT a buffer (BPTT-connected)

    def reset(self) -> None:
        """Clear the carried psi_{k-1}. Call at the start of every rollout (window/sim)."""
        self._psi_prev = None

    def _psi(self, z: Tensor) -> Tensor:
        # bounded map: psi = psi_scale * tanh(net(z)) in (-psi_scale, psi_scale)
        assert z.size(1) == self.nz
        w = self.net(z.view(-1, self.nz))
        return self.psi_scale * torch.tanh(w).view(-1, self.nw, 1)

    def forward(self, z: Tensor) -> Tensor:
        psi = self._psi(z)                                   # (batch, nw, 1), bounded
        prev = self._psi_prev
        if prev is None or prev.shape[0] != psi.shape[0]:
            # first step of the rollout (or batch changed): no correction, seed psi_{-1} := psi_0
            g = torch.zeros_like(psi)
        else:
            g = psi - prev
        self._psi_prev = psi                                 # carry (live tensor -> BPTT connects)
        return g
