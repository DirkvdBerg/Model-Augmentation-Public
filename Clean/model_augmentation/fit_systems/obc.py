"""One-step orthogonal-by-construction (OBC) augmentation: the GENERIC part (D-192)."""
__project_origin__ = "added"

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class OBCRankLossAbort(RuntimeError):
    """Raised when the reference basis lost rank at two consecutive refreshes (or at the first
    build, when there is no previous valid basis to retain)."""


class OBCBasis:
    """Economy SVD of the stacked baseline sensitivity, frozen for one epoch."""

    def __init__(self, J: Tensor, vbar: Tensor):
        """Factorise a float64 `J (n_rows, n_cols)` with its columns scaled to unit norm, `no_grad`
        throughout. `vbar` is the detached float64 expansion point of `J`."""
        with torch.no_grad():
            n_rows, n_cols = J.shape
            G_raw = J.T @ J
            norms_raw = G_raw.diagonal().clamp_min(0).sqrt()
            column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
            U, S, Vh = torch.linalg.svd(J / column_scale, full_matrices=False)
            # THEORY: numpy.linalg.matrix_rank default, S.max() * max(M, N) * eps(dtype)
            # (Golub and Van Loan, Matrix Computations, 4th ed., sect. 5.4.1 numerical rank).
            tol = float(S[0]) * (max(n_rows, n_cols) * torch.finfo(S.dtype).eps)
            rank = int((S > tol).sum())
            if rank < n_cols:
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
        self.U, self.S, self.Vh = U, S, Vh
        self.column_scale = column_scale
        self.vbar = vbar
        self.n_cols, self.rank, self.tol = n_cols, rank, tol

    def coefficient(self, F: Tensor) -> Tensor:
        """`theta = J^+ F = V S^-1 U^T F`, float64, differentiable in `F` (any float dtype)."""
        F64 = F.to(torch.float64)
        theta = self.Vh.T @ ((self.U.T @ F64) / self.S)
        return theta / self.column_scale

    def to(self, device) -> 'OBCBasis':
        self.U, self.S, self.Vh = self.U.to(device), self.S.to(device), self.Vh.to(device)
        self.vbar = self.vbar.to(device)
        self.column_scale = self.column_scale.to(device)
        return self

    @property
    def device(self):
        return self.U.device


def build_stacked_sensitivity(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor, Z: Tensor,
                              row_ix: Sequence[int], chunk: int = 16384,
                              device=None) -> Tensor:
    """`J (N * n_rows_sel, p)` in float64: `d step(v, z_k) / dv` at `vbar`, stacked sample-major."""
    if vbar.dtype != torch.float64:
        raise TypeError('build in float64: vbar is %s' % vbar.dtype)
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    vbar = vbar.detach().to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            Jc = torch.func.jacfwd(lambda v: step(v, Zc))(vbar)
            Jc = Jc[:, row_ix, 0, :]
            out.append(Jc.reshape(-1, vbar.shape[0]).detach())
    return torch.cat(out, 0)


def evaluate_step_rows(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor, Z: Tensor,
                       row_ix: Sequence[int], chunk: int, device) -> Tensor:
    """`step(vbar, z_k)` restricted to `row_ix`, stacked sample-major (float64, no grad)."""
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    vbar = vbar.detach().to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            out.append(step(vbar, Zc)[:, row_ix, 0].reshape(-1))
    return torch.cat(out, 0)


class OBCLifecycle:
    """Once-per-objective coefficient and the epoch-refresh policy."""

    max_consecutive_rank_loss = 2

    def __init__(self, reference_field: Callable, basis_builder: Callable):
        self.reference_field = reference_field
        self.basis_builder = basis_builder

        self.basis: Optional[OBCBasis] = None
        self.last_refresh_epoch: Optional[int] = None
        self.consecutive_rank_loss = 0

    def refresh(self, vbar: Tensor, epoch: Optional[int] = None) -> None:
        """Rebuild the basis at a DETACHED copy of `vbar`; apply the rank-loss policy."""
        vbar64 = vbar.detach().to(torch.float64).clone()
        J, _ = self.basis_builder(vbar64)
        cand = OBCBasis(J, vbar64)
        del J
        if self.basis is not None and cand.device != self.basis.device:
            cand.to(self.basis.device)
        if cand.rank < cand.n_cols:
            self.consecutive_rank_loss += 1
            msg = ('[obc] refresh at epoch %s: numerical rank %d < %d (tol %.3e). '
                   % (epoch, cand.rank, cand.n_cols, cand.tol))
            if self.basis is None:
                raise OBCRankLossAbort(msg + 'No previous valid basis exists to retain; the '
                                       'reference set does not separate the protected '
                                       'directions at this parameter point.')
            if self.consecutive_rank_loss >= self.max_consecutive_rank_loss:
                raise OBCRankLossAbort(msg + '%d consecutive refreshes lost rank; OBC training '
                                       'readiness is aborted rather than continued on a '
                                       'truncated basis.' % self.consecutive_rank_loss)
        else:
            self.consecutive_rank_loss = 0
            self.basis = cand
        self.last_refresh_epoch = epoch

    def maybe_refresh(self, epoch: int, vbar_fn: Callable[[], Tensor]) -> None:
        """Refresh at the first objective and at the first objective of every new epoch."""
        if self.basis is None or epoch != self.last_refresh_epoch:
            self.refresh(vbar_fn(), epoch)

    def coefficient(self, learning_component) -> Tensor:
        """`theta_aux` for THIS objective evaluation: the current learning component on the
        full reference set, solved in float64 with autograd."""
        F = self.reference_field(learning_component)
        if F.device != self.basis.device:
            self.basis.to(F.device)
        return self.basis.coefficient(F)
