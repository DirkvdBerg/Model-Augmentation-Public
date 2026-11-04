"""Fit the network correction to the baseline derivatives and next-state values, then subtract the fit.

On fixed normalised reference tuples, Phi stacks the baseline derivatives with respect to its trained parameters
beta at beta_lin (beta at the start of the epoch),
F_base stacks the baseline next states, and F stacks the network's physical-state corrections:
    J = [Phi, F_base]
    theta_aux = argmin_theta ||F - J theta||^2
    F_projected = F - J theta_aux, so J^T F_projected = 0.
Orthogonality holds over the stacked tuples at zero added states, not at each tuple separately.
obc_gantry.py supplies the reference tuples and applies the correction during simulation.
"""
__project_origin__ = "added"

from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class OBCBasis:
    """J in factorised form (SVD), so theta_aux can be refitted cheaply at every update."""

    def __init__(self, J: Tensor, beta_lin: Tensor):
        """Factorise J D^-1 = U diag(S) V^T; D contains column norms (1 for zero columns).

        Scaling prevents column size from dominating the rank test. Discard singular values at or below the numerical
        rank threshold; OBCProjection rejects a basis with dependent columns.

        Args:
            J: (M, q) float64 matrix: M stacked state values, q columns; (6*N, 11) for N gantry tuples.
            beta_lin: baseline coordinates used to build J; a vector of length 10 for the gantry.
        """
        with torch.no_grad():
            n_rows, n_cols = J.shape
            gram = J.T @ J
            column_norms = gram.diagonal().sqrt()
            column_scale = torch.where(column_norms > 0, column_norms, torch.ones_like(column_norms))
            U, S, Vh = torch.linalg.svd(J / column_scale, full_matrices=False)
            # Numerical rank: S_i > S_max * max(rows, columns) * machine precision, the numpy matrix_rank default.
            tol = float(S[0]) * (max(n_rows, n_cols) * torch.finfo(S.dtype).eps)
            rank = int((S > tol).sum())
            if rank < n_cols:
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
        self.U, self.S, self.Vh = U, S, Vh
        self.column_scale = column_scale
        self.beta_lin = beta_lin
        self.n_cols, self.rank, self.tol = n_cols, rank, tol

    def coefficient(self, F: Tensor) -> Tensor:
        """Least-squares fit of F on J: theta_aux = D^-1 V diag(1 / S) U^T F.

        For full-column-rank J, this equals (J^T J)^-1 J^T F without forming the inverse.
        If singular values were discarded, the fit uses only the directions of the remaining singular vectors.

        Args:
            F: (M,) network corrections in the same row order as J.
        Returns:
            (q,) float64 coefficients, one per column of J.
        """
        F64 = F.to(torch.float64)
        # Undo column scaling to recover coefficients of J.
        theta = self.Vh.T @ ((self.U.T @ F64) / self.S)
        return theta / self.column_scale

    def to(self, device) -> 'OBCBasis':
        """Move stored tensors to device in place; return self."""
        self.U, self.S, self.Vh = self.U.to(device), self.S.to(device), self.Vh.to(device)
        self.beta_lin = self.beta_lin.to(device)
        self.column_scale = self.column_scale.to(device)
        return self

    @property
    def device(self):
        """Device of the stored factors."""
        return self.U.device


def build_stacked_sensitivity(step: Callable[[Tensor, Tensor], Tensor], beta_lin: Tensor, Z: Tensor,
                              physical_rows: Sequence[int], chunk: int,
                              device=None, append_value: bool = False) -> Tensor:
    """Build Phi: the baseline's sensitivity to beta at every tuple, varphi_i = d step(beta, z_i) / d beta at beta_lin.

    Rows follow tuple order, then physical_rows order. With append_value, return J = [Phi, F_base].
    The extra column includes the baseline prediction in the projection. The thesis uses the offset
    Gamma = F_base - Phi beta_lin; [Phi, Gamma] spans the same space. In the code, theta_aux = (a, c),
    where a weights Phi and c weights F_base; the thesis coefficients are (a + c beta_lin, c).

    Args:
        step: function step(beta, Z_batch) returning next states, shape (batch, states, 1).
            Each tuple is evaluated independently.
        beta_lin: (p,) float64 baseline coordinates at which derivatives are computed; p = 10 for the gantry.
        Z: (N, d, 1) reference tuples: N samples, d state/input values (9 for the gantry).
        physical_rows: state indices to retain; r = len(physical_rows).
        chunk: tuples processed together, to limit memory use.
        device: build device; defaults to beta_lin.device.
        append_value: append baseline next-state values as the last column.
    Returns:
        Float64 Phi of shape (N*r, p), or J of shape (N*r, p+1) with append_value.
    """
    if beta_lin.dtype != torch.float64:
        raise TypeError('build in float64: beta_lin is %s; pass beta_lin.double()' % beta_lin.dtype)
    dev = device if device is not None else beta_lin.device
    physical_rows = torch.as_tensor(list(physical_rows), dtype=torch.long, device=dev)
    beta_lin = beta_lin.detach().to(dev)
    out, values = [], []
    with torch.no_grad():
        # Use forward-mode differentiation: few parameters, many reference-state outputs.
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            Jc = torch.func.jacfwd(lambda v: step(v, Zc))(beta_lin)
            out.append(Jc[:, physical_rows, 0, :].reshape(-1, beta_lin.shape[0]))
            if append_value:
                values.append(step(beta_lin, Zc)[:, physical_rows, 0].reshape(-1))
    J = torch.cat(out, 0)
    if append_value:
        J = torch.cat((J, torch.cat(values, 0)[:, None]), dim=1)
    return J


class OBCProjection:
    """Refresh the basis once per epoch and refit the network correction at each loss evaluation.

    Epoch updates follow the baseline parameters without rebuilding J at every evaluation.
    Training differentiates the fitted correction with respect to the network parameters while keeping J fixed.
    """

    # Keep the previous full-rank basis after one rank-deficient refresh; abort after two in a row.
    max_consecutive_rank_loss = 2

    def __init__(self, reference_field: Callable, basis_builder: Callable):
        """Set the two functions used by the projection; call refresh before coefficient.

        Args:
            reference_field: function taking the network block and returning the stacked correction F.
            basis_builder: function taking beta_lin and returning J, with rows matching F.
        """
        self.reference_field = reference_field
        self.basis_builder = basis_builder

        self.basis: Optional[OBCBasis] = None
        self.last_refresh_epoch: Optional[int] = None
        self.consecutive_rank_loss = 0

    def refresh(self, beta_lin: Tensor, epoch: int) -> None:
        """Build J at beta_lin and accept it only at full numerical column rank.

        Dependent columns give non-unique coefficients. Keep the previous basis and its beta_lin on rank loss;
        raise RuntimeError if no previous basis exists or the consecutive-rank-loss limit is reached.
        Record epoch after an accepted update or fallback; acceptance resets the rank-loss count.
        """
        beta_lin64 = beta_lin.detach().to(torch.float64).clone()
        J = self.basis_builder(beta_lin64)
        candidate_basis = OBCBasis(J, beta_lin64)
        del J
        if self.basis is not None and candidate_basis.device != self.basis.device:
            candidate_basis.to(self.basis.device)
        if candidate_basis.rank < candidate_basis.n_cols:
            self.consecutive_rank_loss += 1
            msg = ('[obc] refresh at epoch %s: numerical rank %d < %d (tol %.3e)'
                   % (epoch, candidate_basis.rank, candidate_basis.n_cols, candidate_basis.tol))
            if self.basis is None:
                raise RuntimeError(msg + ' at the first build, with no previous basis to keep; check that the '
                                   'reference tuples excite every training coordinate, and the linear '
                                   'dependence among the sensitivity columns and the baseline column')
            if self.consecutive_rank_loss >= self.max_consecutive_rank_loss:
                raise RuntimeError(msg + ' for %d refreshes in a row; check reference excitation and linear '
                                   'dependence among the sensitivity columns and the baseline column at this '
                                   'parameter point' % self.consecutive_rank_loss)
        else:
            self.consecutive_rank_loss = 0
            self.basis = candidate_basis
        self.last_refresh_epoch = epoch

    def maybe_refresh(self, epoch: int, beta: Tensor) -> None:
        """Request a basis update at the current beta once per epoch; refresh handles rank loss."""
        if self.basis is None or epoch != self.last_refresh_epoch:
            self.refresh(beta, epoch)

    def coefficient(self, learning_component) -> Tensor:
        """theta_aux for the current network: its correction F at the tuples, fitted on the stored J.

        Training accounts for how both F and the fitted part J theta_aux change with the network parameters.

        Returns:
            Float64 coefficients, one per column of J (11 for the gantry).
        """
        F = self.reference_field(learning_component)
        if F.device != self.basis.device:
            self.basis.to(F.device)
        return self.basis.coefficient(F)
