"""Orthogonal projection of the network output onto the baseline sensitivity and offset directions (Gyorok et al.),
the model-independent part: the basis factorisation and the coefficient theta_aux; the gantry code supplies J."""
__project_origin__ = "added"

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class OBCRankLossAbort(RuntimeError):
    """The projection basis lost column rank at the first build, or at too many refreshes in a row."""


class OBCBasis:
    """Economy SVD of a column-scaled projection basis J, with its numerical rank."""

    def __init__(self, J: Tensor, vbar: Tensor):
        """J (n_rows, n_cols) in float64; vbar, the detached float64 point beta_lin it is taken at. A rank below
        n_cols is recorded and the factors truncated, so the caller can reject the basis."""
        with torch.no_grad():
            n_rows, n_cols = J.shape
            G_raw = J.T @ J
            norms_raw = G_raw.diagonal().clamp_min(0).sqrt()
            column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
            U, S, Vh = torch.linalg.svd(J / column_scale, full_matrices=False)
            # THEORY: numerical rank of the column-scaled J, tol = S_max max(M, N) eps (Golub and Van Loan; the
            # numpy.linalg.matrix_rank default)
            tol = float(S[0]) * (max(n_rows, n_cols) * torch.finfo(S.dtype).eps)
            rank = int((S > tol).sum())
            if rank < n_cols:
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
        self.U, self.S, self.Vh = U, S, Vh
        self.column_scale = column_scale
        self.vbar = vbar
        self.n_cols, self.rank, self.tol = n_cols, rank, tol

    def coefficient(self, F: Tensor) -> Tensor:
        """theta_aux = D^-1 V S^-1 U^T F with J D^-1 = U S V^T, D = diag(column_scale): for a full-rank J this is
        (J^T J)^-1 J^T F. In float64, differentiable in F."""
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
    """Phi: d step(beta, z_i) / d beta at beta_lin = vbar on the rows row_ix, stacked sample-major, float64."""
    if vbar.dtype != torch.float64:
        raise TypeError('build in float64: vbar is %s; pass vbar.double()' % vbar.dtype)
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
    """The transition step(vbar, z_i) on the rows row_ix, stacked sample-major, float64 without gradient."""
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    vbar = vbar.detach().to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            out.append(step(vbar, Zc)[:, row_ix, 0].reshape(-1))
    return torch.cat(out, 0)


class OBCProjection:
    """The projection basis with its rank policy and refresh, and the coefficient theta_aux of the network."""

    max_consecutive_rank_loss = 2

    def __init__(self, reference_field: Callable, basis_builder: Callable):
        """reference_field: network -> F, its stacked output on the reference tuples; basis_builder: vbar ->
        (J, seconds)."""
        self.reference_field = reference_field
        self.basis_builder = basis_builder

        self.basis: Optional[OBCBasis] = None
        self.last_refresh_epoch: Optional[int] = None
        self.consecutive_rank_loss = 0

    def refresh(self, vbar: Tensor, epoch: Optional[int] = None) -> None:
        """Rebuild the basis at a detached copy of vbar; a basis without full column rank keeps the previous one,
        and raises at the first build or after max_consecutive_rank_loss refreshes in a row."""
        vbar64 = vbar.detach().to(torch.float64).clone()
        J, _ = self.basis_builder(vbar64)
        candidate_basis = OBCBasis(J, vbar64)
        del J
        if self.basis is not None and candidate_basis.device != self.basis.device:
            candidate_basis.to(self.basis.device)
        if candidate_basis.rank < candidate_basis.n_cols:
            self.consecutive_rank_loss += 1
            msg = ('[obc] refresh at epoch %s: numerical rank %d < %d (tol %.3e)'
                   % (epoch, candidate_basis.rank, candidate_basis.n_cols, candidate_basis.tol))
            if self.basis is None:
                raise OBCRankLossAbort(msg + ' at the first build, with no previous basis to keep; check that the '
                                       'reference tuples excite every training coordinate and that the protected '
                                       'rows and the baseline column are not redundant')
            if self.consecutive_rank_loss >= self.max_consecutive_rank_loss:
                raise OBCRankLossAbort(msg + ' for %d refreshes in a row; check the reference excitation and the '
                                       'protected directions at this parameter point'
                                       % self.consecutive_rank_loss)
        else:
            self.consecutive_rank_loss = 0
            self.basis = candidate_basis
        self.last_refresh_epoch = epoch

    def maybe_refresh(self, epoch: int, vbar_fn: Callable[[], Tensor]) -> None:
        """Refresh at the first objective and at the first objective of every new epoch."""
        if self.basis is None or epoch != self.last_refresh_epoch:
            self.refresh(vbar_fn(), epoch)

    def coefficient(self, learning_component) -> Tensor:
        """theta_aux for this objective: the network on all reference tuples, solved on the fixed basis and
        differentiable through F."""
        F = self.reference_field(learning_component)
        if F.device != self.basis.device:
            self.basis.to(F.device)
        return self.basis.coefficient(F)
