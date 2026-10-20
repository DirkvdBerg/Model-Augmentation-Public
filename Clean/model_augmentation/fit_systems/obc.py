"""One-step orthogonal-by-construction (OBC) augmentation: the GENERIC part (D-192)."""
__project_origin__ = "added"

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class OBCRankLossAbort(RuntimeError):
    """Raised when the reference basis lost rank at two consecutive refreshes (or at the first
    build, when there is no previous valid basis to retain)."""


@dataclass
class RankReport:
    """Everything a refresh must report about the stacked sensitivity `J`."""
    n_rows: int
    n_cols: int
    singular_values: list
    rank: int
    tol: float
    cond: float


class OBCBasis:
    """Economy SVD of the stacked baseline sensitivity, frozen for one epoch."""

    def __init__(self, U: Tensor, S: Tensor, Vh: Tensor, vbar: Tensor, report: RankReport,
                 column_scale: Optional[Tensor] = None):
        self.U, self.S, self.Vh = U, S, Vh
        self.column_scale = column_scale
        self.vbar = vbar.detach().clone()
        self.report = report
        self.n_rows, self.n_cols = report.n_rows, report.n_cols
        self.rank = report.rank

    @staticmethod
    def rank_tolerance(S: Tensor, n_rows: int, n_cols: int, rtol: Optional[float] = None) -> float:
        # THEORY: numpy.linalg.matrix_rank default, S.max() * max(M, N) * eps(dtype)
        # (Golub and Van Loan, Matrix Computations, 4th ed., sect. 5.4.1 numerical rank).
        if rtol is None:
            rtol = max(n_rows, n_cols) * torch.finfo(S.dtype).eps
        return float(S[0]) * float(rtol)

    @classmethod
    def from_matrix(cls, J: Tensor, vbar: Tensor, rtol: Optional[float] = None,
                    scale_columns: bool = False) -> 'OBCBasis':
        """Factorise a float64 `J (n_rows, n_cols)`. `no_grad` throughout."""
        with torch.no_grad():
            n_rows, n_cols = J.shape
            G_raw = J.T @ J
            norms_raw = G_raw.diagonal().clamp_min(0).sqrt()
            column_scale = None
            J_factor = J
            if scale_columns:
                column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
                J_factor = J / column_scale
            U, S, Vh = torch.linalg.svd(J_factor, full_matrices=False)
            tol = cls.rank_tolerance(S, n_rows, n_cols, rtol)
            rank = int((S > tol).sum())
            cond = float(S[0] / S[rank - 1]) if rank > 0 else float('inf')
            report = RankReport(
                n_rows=n_rows, n_cols=n_cols,
                singular_values=[float(s) for s in S],
                rank=rank, tol=tol, cond=cond)
            if rank < n_cols:
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
        return cls(U, S, Vh, vbar, report, column_scale=column_scale)

    def coefficient(self, F: Tensor) -> Tensor:
        """`theta = J^+ F = V S^-1 U^T F`, float64, differentiable in `F` (any float dtype)."""
        F64 = F.to(torch.float64)
        theta = self.Vh.T @ ((self.U.T @ F64) / self.S)
        return theta if self.column_scale is None else theta / self.column_scale

    def to(self, device) -> 'OBCBasis':
        self.U, self.S, self.Vh = self.U.to(device), self.S.to(device), self.Vh.to(device)
        self.vbar = self.vbar.to(device)
        if self.column_scale is not None:
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
                       row_ix: Sequence[int], chunk: int = 16384, device=None) -> Tensor:
    """`step(vbar, z_k)` restricted to `row_ix`, stacked sample-major (float64, no grad)."""
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    out = []
    with torch.no_grad():
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            out.append(step(vbar.detach().to(dev), Zc)[:, row_ix, 0].reshape(-1))
    return torch.cat(out, 0)


class OBCLifecycle:
    """Once-per-objective coefficient and the epoch-refresh policy."""

    def __init__(self, reference_field: Callable, basis_builder: Callable, refresh: str = 'epoch',
                 rtol: Optional[float] = None, out_dtype: torch.dtype = torch.float32,
                 expected_rank: Optional[int] = None, max_consecutive_rank_loss: int = 2,
                 scale_columns: bool = False):
        self.reference_field = reference_field
        self.basis_builder = basis_builder
        self.refresh_policy = refresh
        self.rtol = rtol
        self.out_dtype = out_dtype
        self.expected_rank = expected_rank
        self.max_consecutive_rank_loss = int(max_consecutive_rank_loss)
        self.scale_columns = bool(scale_columns)

        self.basis: Optional[OBCBasis] = None
        self.last_refresh_epoch: Optional[int] = None
        self.n_refresh = 0
        self.n_rank_loss = 0
        self.consecutive_rank_loss = 0
        self.timing = {'field_and_solve_s': [], 'refresh_s': []}

    def refresh(self, vbar: Tensor, epoch: Optional[int] = None) -> RankReport:
        """Rebuild the basis at a DETACHED copy of `vbar`; apply the rank-loss policy."""
        t0 = time.perf_counter()
        vbar64 = vbar.detach().to(torch.float64).clone()
        J, build_s = self.basis_builder(vbar64)
        cand = OBCBasis.from_matrix(J, vbar64, rtol=self.rtol,
                                    scale_columns=self.scale_columns)
        del J
        if self.basis is not None and cand.device != self.basis.device:
            cand.to(self.basis.device)
        rep = cand.report
        expected = self.expected_rank if self.expected_rank is not None else rep.n_cols
        entry = {'epoch': epoch, 'rank': rep.rank, 'expected': expected, 'cond': rep.cond,
                 'singular_values': rep.singular_values, 'accepted': rep.rank >= expected}
        if rep.rank < expected:
            self.n_rank_loss += 1
            self.consecutive_rank_loss += 1
            msg = ('[obc] refresh at epoch %s: numerical rank %d < %d (tol %.3e). '
                   % (epoch, rep.rank, expected, rep.tol))
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
        self.n_refresh += 1
        self.last_refresh_epoch = epoch
        entry['seconds'] = time.perf_counter() - t0
        self.timing['refresh_s'].append(entry['seconds'])
        return rep

    def maybe_refresh(self, epoch: int, vbar_fn: Callable[[], Tensor]) -> bool:
        """Refresh when the policy says so. Returns True when a refresh ran."""
        if self.basis is None:
            self.refresh(vbar_fn(), epoch)
            return True
        if self.refresh_policy == 'epoch' and epoch != self.last_refresh_epoch:
            self.refresh(vbar_fn(), epoch)
            return True
        return False

    def coefficient(self, learning_component) -> Tensor:
        """`theta_aux` for THIS objective evaluation: the current learning component on the
        full reference set, solved in float64 with autograd, cast to the pipeline dtype."""
        F = self.reference_field(learning_component)
        if F.device != self.basis.device:
            self.basis.to(F.device)
        theta64 = self.basis.coefficient(F)
        return theta64.to(self.out_dtype)

    def prediction_coefficient(self, learning_component) -> Tensor:
        """Current coefficient for validation/prediction, without training bookkeeping."""
        with torch.no_grad():
            F = self.reference_field(learning_component)
            if F.device != self.basis.device:
                self.basis.to(F.device)
            theta64 = self.basis.coefficient(F)
            theta = theta64.to(self.out_dtype)
            return theta
