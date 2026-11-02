"""Orthogonal-by-construction projection (Gyorok et al.), model-independent part: the basis J of what the baseline
parameters can change, and the coefficient theta_aux of the network field on it; obc_gantry.py supplies J and F."""
__project_origin__ = "added"

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class OBCRankLossAbort(RuntimeError):
    """The projection basis lost column rank at the first build, or at too many refreshes in a row."""


class OBCBasis:
    """The basis J factorised once: economy SVD of J with unit-norm columns, and its numerical rank."""

    def __init__(self, J: Tensor, vbar: Tensor):
        """Factorise J D^-1 = U S V^T, D = column norms of J; a rank loss truncates the factors.

        Args:
            J: (M, 11) basis [Phi F_base^n(beta_lin)], float64, M = N r for r protected rows.
            vbar: (10,) beta_lin, float64.
        """
        with torch.no_grad():
            n_rows, n_cols = J.shape
            # 1. Column scale, so columns of very different size get equal weight in the rank test:
            #    D = diag(||J_:,i||), 1 for a zero column.
            G_raw = J.T @ J
            norms_raw = G_raw.diagonal().sqrt()
            column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
            # 2. Factorise once; every later coefficient is then a cheap product: J D^-1 = U S V^T.
            U, S, Vh = torch.linalg.svd(J / column_scale, full_matrices=False)
            # 3. Numerical rank: singular values above tol count; on rank loss keep only those directions.
            # THEORY: tol = S_max max(M, N) eps (Golub and Van Loan; the numpy.linalg.matrix_rank default)
            tol = float(S[0]) * (max(n_rows, n_cols) * torch.finfo(S.dtype).eps)
            rank = int((S > tol).sum())
            if rank < n_cols:
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
        self.U, self.S, self.Vh = U, S, Vh
        self.column_scale = column_scale
        self.vbar = vbar
        self.n_cols, self.rank, self.tol = n_cols, rank, tol

    def coefficient(self, F: Tensor) -> Tensor:
        """OBC coefficient: the least-squares fit of F on the columns of J, so F - J theta_aux is orthogonal to J.

        theta_aux = D^-1 V S^-1 U^T F = (J^T J)^-1 J^T F for a full-rank J, differentiable in F.

        Args:
            F: (M,) stacked network field F^n_aug, ordered as the rows of J.
        Returns:
            (11,) theta_aux = (a, c), float64.
        """
        F64 = F.to(torch.float64)
        # solve with the stored factors, then undo the column scale: theta_aux = D^-1 V S^-1 U^T F
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
                              row_ix: Sequence[int], chunk: int,
                              device=None, append_value: bool = False) -> Tensor:
    """How the baseline step moves when its parameters move, stacked over the reference tuples.

    Phi = col(varphi(beta_lin, z_i)), varphi = d step(beta, z_i) / d beta at beta_lin, rows row_ix, tuple by tuple.

    Args:
        step: (beta, Z) -> next state, e.g. f_base^n(theta_base(beta), z).
        vbar: (10,) beta_lin, float64.
        Z: (N, 9, 1) tuples (x_tilde^n_i, u^n_i).
        row_ix: protected rows, the state rows the projection keeps orthogonal.
        chunk: tuples per jacfwd call; sets the memory use, not the result.
        device: device of the build, else that of vbar.
        append_value: add step(beta_lin, z_i) as a last column (the f_base^n column of J).
    Returns:
        (N r, p) Phi, or (N r, p + 1) J with append_value, float64; r = len(row_ix), p = vbar.numel().
    """
    # 1. Build in float64, on the chosen device.
    if vbar.dtype != torch.float64:
        raise TypeError('build in float64: vbar is %s; pass vbar.double()' % vbar.dtype)
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    vbar = vbar.detach().to(dev)
    out, values = [], []
    with torch.no_grad():
        # 2. Parameter Jacobian of the step, per chunk of tuples (forward mode: 10 parameters, many outputs).
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            # varphi_i = d step(beta, z_i) / d beta at beta_lin, rows row_ix
            Jc = torch.func.jacfwd(lambda v: step(v, Zc))(vbar)
            out.append(Jc[:, row_ix, 0, :].reshape(-1, vbar.shape[0]))
            if append_value:
                values.append(step(vbar, Zc)[:, row_ix, 0].reshape(-1))
    # 3. Stack the chunks; optionally append the baseline step itself as the last column.
    J = torch.cat(out, 0)
    if append_value:
        # J = [Phi  col(step(beta_lin, z_i))]
        J = torch.cat((J, torch.cat(values, 0)[:, None]), dim=1)
    return J


class OBCProjection:
    """The basis J, rebuilt at beta_lin once per epoch with a rank check, and theta_aux of the network on it."""

    # HEURISTIC: a rank-deficient refresh keeps the last basis; two in a row abort the run
    max_consecutive_rank_loss = 2

    def __init__(self, reference_field: Callable, basis_builder: Callable):
        """Projection with no basis yet; the first refresh builds it.

        Args:
            reference_field: network block -> (M,) its field F^n_aug on the reference tuples, ordered as J's rows.
            basis_builder: beta_lin -> (J, build time [s]).
        """
        self.reference_field = reference_field
        self.basis_builder = basis_builder

        # current basis; epoch of the last refresh; rank-deficient refreshes in a row
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
        # 1. Build a candidate basis at the new beta_lin, on the device of the current one.
        vbar64 = vbar.detach().to(torch.float64).clone()
        J, _ = self.basis_builder(vbar64)
        candidate_basis = OBCBasis(J, vbar64)
        del J
        if self.basis is not None and candidate_basis.device != self.basis.device:
            candidate_basis.to(self.basis.device)
        # 2. Accept it at full rank; on rank loss keep the last basis, or abort if there is none or the limit is hit.
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

        beta_lin = beta at the first objective of every epoch, fixed within the epoch.

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
