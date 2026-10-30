"""Orthogonal-by-construction projection (Gyorok et al.), the model-independent part: the factorised basis J and
the coefficient theta_aux = (J^T J)^-1 J^T F^n_aug; the gantry code supplies J and F. J = [Phi F_base^n(beta_lin)]
spans the same space as the thesis Phi_Gamma, but theta_aux = (a, c) on J is (a + beta_lin c, c) on Phi_Gamma.
"""
__project_origin__ = "added"

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class OBCRankLossAbort(RuntimeError):
    """The projection basis lost column rank at the first build, or at too many refreshes in a row."""


class OBCBasis:
    """J D^-1 = U S V^T, D_ii = ||J_:,i|| (1 for a zero column): the economy SVD of the column-scaled basis, with its
    numerical rank."""

    def __init__(self, J: Tensor, vbar: Tensor):
        """Factorised OBC basis.

        J D^-1 = U S V^T, D_ii = ||J_:,i|| (1 for a zero column), with numerical rank at tol = S_max max(M, N) eps;
        truncated factors mark a rank loss for the caller to reject.

        Args:
            J: (M, 11) basis [Phi F_base^n(beta_lin)], float64, M = N r for r protected physical rows (r = 6 in the
                thesis configuration).
            vbar: (10,) beta_lin, detached float64.
        """
        with torch.no_grad():
            n_rows, n_cols = J.shape
            # 1. Column scale D = diag(||J_:,i||), 1 for a zero column.
            G_raw = J.T @ J
            norms_raw = G_raw.diagonal().sqrt()
            column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
            # 2. Factorise.
            # J D^-1 = U S V^T.
            U, S, Vh = torch.linalg.svd(J / column_scale, full_matrices=False)
            # 3. Numerical rank; truncate on rank loss.
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
        """OBC coefficient theta_aux.

        theta_aux = D^-1 V S^-1 U^T F = (J^T J)^-1 J^T F for a full-rank J, differentiable in F; F - J theta_aux is then
        orthogonal to every column of J.

        Args:
            F: (M,) stacked network field F^n_aug, in the sample and row order of J.
        Returns:
            (11,) theta_aux = (a, c), float64.
        """
        F64 = F.to(torch.float64)
        # theta_aux = D^-1 V S^-1 U^T F.
        theta = self.Vh.T @ ((self.U.T @ F64) / self.S)
        return theta / self.column_scale

    def to(self, device) -> 'OBCBasis':
        """Move the factors to device."""
        self.U, self.S, self.Vh = self.U.to(device), self.S.to(device), self.Vh.to(device)
        self.vbar = self.vbar.to(device)
        self.column_scale = self.column_scale.to(device)
        return self

    @property
    def device(self):
        """Device of the factors."""
        return self.U.device


def build_stacked_sensitivity(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor, Z: Tensor,
                              row_ix: Sequence[int], chunk: int = 16384,
                              device=None, append_value: bool = False) -> Tensor:
    """Stacked baseline sensitivity Phi.

    Phi = col(varphi(beta_lin, z_i)), varphi = d step(beta, z_i) / d beta at beta_lin, on the rows row_ix, stacked
    sample-major.

    Args:
        step: (beta, Z) -> next state, e.g. f_base^n(theta_base(beta), z).
        vbar: (10,) beta_lin, float64.
        Z: (N, 9, 1) tuples (x_tilde^n_i, u^n_i).
        row_ix: protected rows.
        chunk: tuples per jacfwd call.
        device: device of the build, else that of vbar.
        append_value: add step(beta_lin, z_i) as a last column (the f_base^n column of J).
    Returns:
        (N r, p) Phi, or (N r, p + 1) J with append_value, float64; r = len(row_ix), p = vbar.numel() (6 and 10
        in the thesis configuration).
    """
    # 1. Prepare.
    if vbar.dtype != torch.float64:
        raise TypeError('build in float64: vbar is %s; pass vbar.double()' % vbar.dtype)
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    vbar = vbar.detach().to(dev)
    out, values = [], []
    with torch.no_grad():
        # 2. Jacobian per chunk of tuples.
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            # varphi_i = d step(beta, z_i) / d beta at beta_lin, rows row_ix, stacked sample-major.
            Jc = torch.func.jacfwd(lambda v: step(v, Zc))(vbar)
            out.append(Jc[:, row_ix, 0, :].reshape(-1, vbar.shape[0]))
            if append_value:
                values.append(step(vbar, Zc)[:, row_ix, 0].reshape(-1))
    # 3. Stack, with the optional f_base^n column.
    J = torch.cat(out, 0)
    if append_value:
        # J = [Phi  col(step(beta_lin, z_i))].
        J = torch.cat((J, torch.cat(values, 0)[:, None]), dim=1)
    return J


class OBCProjection:
    """The basis J at beta_lin with its rank policy and per-epoch refresh, and theta_aux of the network."""

    max_consecutive_rank_loss = 2

    def __init__(self, reference_field: Callable, basis_builder: Callable):
        """Store the reference-field evaluator and the basis builder returning (J, seconds)."""
        self.reference_field = reference_field
        self.basis_builder = basis_builder

        self.basis: Optional[OBCBasis] = None
        self.last_refresh_epoch: Optional[int] = None
        self.consecutive_rank_loss = 0

    def refresh(self, vbar: Tensor, epoch: int) -> None:
        """Rebuild the OBC basis at beta_lin.

        J at beta_lin = vbar (detached); a rank-deficient basis keeps the last full-rank one, up to the abort limit.

        Args:
            vbar: (10,) beta, the new beta_lin.
            epoch: integer epoch of the refresh.
        """
        # 1. Candidate basis at beta_lin.
        vbar64 = vbar.detach().to(torch.float64).clone()
        J, _ = self.basis_builder(vbar64)
        candidate_basis = OBCBasis(J, vbar64)
        del J
        if self.basis is not None and candidate_basis.device != self.basis.device:
            candidate_basis.to(self.basis.device)
        # 2. Rank policy: keep the last full-rank basis or abort on rank loss, else accept.
        if candidate_basis.rank < candidate_basis.n_cols:
            self.consecutive_rank_loss += 1
            msg = ('[obc] refresh at epoch %s: numerical rank %d < %d (tol %.3e)'
                   % (epoch, candidate_basis.rank, candidate_basis.n_cols, candidate_basis.tol))
            if self.basis is None:
                raise OBCRankLossAbort(msg + ' at the first build, with no previous basis to keep; check that the '
                                       'reference tuples excite every training coordinate, and the linear '
                                       'dependence among the sensitivity columns and the baseline column')
            if self.consecutive_rank_loss >= self.max_consecutive_rank_loss:
                raise OBCRankLossAbort(msg + ' for %d refreshes in a row; check reference excitation and linear '
                                       'dependence among the sensitivity columns and the baseline column at this '
                                       'parameter point' % self.consecutive_rank_loss)
        else:
            self.consecutive_rank_loss = 0
            self.basis = candidate_basis
        self.last_refresh_epoch = epoch

    def maybe_refresh(self, epoch: int, vbar_fn: Callable[[], Tensor]) -> None:
        """Per-epoch refresh of beta_lin.

        beta_lin = beta at the first objective and at the first objective of every epoch, fixed within the epoch.

        Args:
            epoch: integer epoch.
            vbar_fn: returns the current beta.
        """
        if self.basis is None or epoch != self.last_refresh_epoch:
            self.refresh(vbar_fn(), epoch)

    def coefficient(self, learning_component) -> Tensor:
        """theta_aux of the current network.

        theta_aux(theta_aug) = (J^T J)^-1 J^T F^n_aug(theta_aug) on the implemented J, differentiable in F.

        Args:
            learning_component: the network block.
        Returns:
            (11,) theta_aux.
        """
        F = self.reference_field(learning_component)
        if F.device != self.basis.device:
            self.basis.to(F.device)
        return self.basis.coefficient(F)
