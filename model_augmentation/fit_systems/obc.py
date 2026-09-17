"""One-step orthogonal-by-construction (OBC) augmentation: the GENERIC part (D-192).

Gyorok's by-construction lifecycle (orthogonal-IO-augm-main/orthogonalx_augm/src.py,
`IO_orthogonal_augmentation.fit`) carried to a discrete state transition whose parameters enter
nonlinearly:

    J        = col_k d step(v, z_k) / dv |_{v = vbar}      stacked sample-major, float64
    J        = U S V^T                                    economy SVD, declared rank tolerance
    theta    = K_J F,  K_J = V S^-1 U^T                   ONCE per objective, differentiable in F
    F_tilde  = F - J theta                                 so that J^T F_tilde = 0 (stacked)

This module knows a `step(v, z) -> x_plus` callable, a reference set `Z`, a row selection, and
a field `F` produced by some learning component on that reference set. It does NOT know what
a gantry is: no parameter names, no `free_params`, no P transform, no controller, no trajectory
files, no absorber states. Those live in `scripts/gantry/gantry_dynamic/obc_gantry.py`.

Three objects:

* `OBCBasis`      the frozen per-epoch factorisation (U, S, Vh, vbar, rank report) and the
                  differentiable coefficient solve plus the generic orthogonality diagnostics.
* `OBCLifecycle`  the once-per-objective coefficient and the epoch-refresh policy (rank loss
                  retains the previous basis; two consecutive rank losses abort).
* free functions  `build_stacked_sensitivity` (the chunked float64 `jacfwd` build) and
                  `directional_correction` (the `jvp` reference used by equivalence tests).

Nothing here is a loss term. The subtraction is applied inside the model step. The current
D-194/D-195 seam turns the coefficient into a `(mats, dmats)` pair once per objective and threads
that pair as `obc_pass` through `Interconnect.forward` and the rollout.

D-200 optionally factors the affine matrix `[J | step(vbar,z)]`.  Column equilibration is opt-in
and the public `OBCBasis` products remain in the original, unscaled coordinates; the ten-column
tangent construction therefore retains its previous arithmetic.
"""
__project_origin__ = "added"

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


class _StaticCopy(torch.autograd.Function):
    """Write `src` into the fixed-address tensor `dst` and return `dst`, keeping the gradient.

    WHY THIS EXISTS (D-199). CUDA graphs replay a recorded sequence of kernel launches and
    therefore need their inputs at STABLE ADDRESSES. D-194 and D-195 thread freshly allocated
    per-pass tensors into the compiled step (ten for the matrix structure, twenty for the OBC
    pass), which was the suspected replay obstacle: job
    83951 measured 525 s then 1472 s per update at nf 400 under `reduce-overhead`, degrading
    rather than settling, while `default` ran at 0.97 s.

    A plain `dst.copy_(src)` would give a stable address and destroy the gradient path that D-192
    requires: the coefficient must reach the ANN weights and the matrix structure must reach the
    physical parameters. This passes the incoming gradient straight back to `src`, so the address
    is fixed and the chain is intact.

    THE ONE HAZARD, and it fails loudly rather than silently. The buffer is written once per
    objective and read by every timestep of that objective's rollout, so a backward must complete
    before the next write. That holds for the Adam path and for the L-BFGS closure, both of which
    finish a backward before the next forward. If it were ever violated, autograd's version
    counter would raise on the saved tensor rather than return a wrong gradient.
    """

    @staticmethod
    def forward(ctx, src, dst):
        with torch.no_grad():
            dst.copy_(src)
        ctx.mark_dirty(dst)
        return dst

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class StaticPassBuffers:
    """Fixed-address storage for one tuple of per-pass tensors.

    Allocated on first use from the shapes and dtypes it is handed, so it needs no knowledge of
    what it holds, and re-allocated if the device or dtype changes (a `.cuda()` between
    objectives). `torch._dynamo.mark_static_address` tells inductor the address is stable. The
    measured D-199 result is performance-neutral; replay was not established.
    """

    def __init__(self, name: str = 'pass'):
        self.name = name
        self._buf = None

    def _alloc(self, tensors):
        self._buf = [torch.empty_like(t) for t in tensors]
        mark = getattr(torch._dynamo, 'mark_static_address', None)
        if mark is not None:
            for b in self._buf:
                mark(b)

    def __call__(self, tensors):
        """Return the same tuple, routed through the fixed buffers, gradients preserved."""
        tensors = tuple(tensors)
        if (self._buf is None or len(self._buf) != len(tensors)
                or any(b.shape != t.shape or b.dtype != t.dtype or b.device != t.device
                       for b, t in zip(self._buf, tensors))):
            self._alloc(tensors)
        # CLEAR LAST OBJECTIVE'S HISTORY FIRST. `_StaticCopy` marks the buffer dirty, so after a
        # pass the buffer carries that pass's grad_fn. Writing into it again would try to rebase
        # onto a graph that `backward()` has already freed, which raises "Trying to backward
        # through the graph a second time". `detach_` drops the history in place, so the storage
        # and therefore the ADDRESS are untouched, which is the whole point of the buffer.
        for b in self._buf:
            b.detach_()
        return tuple(_StaticCopy.apply(t, b) for t, b in zip(tensors, self._buf))

    def reset(self):
        self._buf = None


class OBCRankLossAbort(RuntimeError):
    """Raised when the reference basis lost rank at two consecutive refreshes (or at the first
    build, when there is no previous valid basis to retain)."""


@dataclass
class RankReport:
    """Everything a refresh must report about the stacked sensitivity `J`."""
    n_rows: int
    n_cols: int
    singular_values: list            # all of them, descending, float
    rank: int
    tol: float                       # absolute tolerance the rank was decided against
    rtol: float                      # tol / S[0]
    cond: float                      # S[0] / S[rank-1]
    col_norms: list                  # ||J[:, j]||
    col_corr: list                   # (n_cols, n_cols) pairwise column correlations
    build_seconds: float = 0.0
    svd_seconds: float = 0.0
    extra: dict = field(default_factory=dict)

    def lines(self, names: Optional[Sequence[str]] = None) -> list:
        names = list(names) if names is not None else [str(j) for j in range(self.n_cols)]
        out = [f'  J: {self.n_rows} x {self.n_cols}, rank {self.rank}/{self.n_cols} at '
               f'tol {self.tol:.3e} (rtol {self.rtol:.3e}), cond {self.cond:.4e}, '
               f'build {self.build_seconds:.1f} s, svd {self.svd_seconds:.1f} s']
        out.append('  singular values: ' + ' '.join(f'{s:.4e}' for s in self.singular_values))
        if self.extra.get('column_scaled'):
            out.append('  SVD coordinates: columns scaled to unit 2-norm; norms below and all '
                       'basis products are in original coordinates')
        out.append('  column norms:    ' + ' '.join(f'{n}={v:.3e}' for n, v in
                                                     zip(names, self.col_norms)))
        worst = []
        for i in range(self.n_cols):
            for j in range(i + 1, self.n_cols):
                worst.append((abs(self.col_corr[i][j]), names[i], names[j], self.col_corr[i][j]))
        worst.sort(reverse=True)
        out.append('  largest |column correlations|: ' + ', '.join(
            f'{a}~{b}={c:+.4f}' for _, a, b, c in worst[:6]))
        return out


class OBCBasis:
    """Economy SVD of the stacked baseline sensitivity, frozen for one epoch.

    Holds `U (n_rows, r)`, `S (r,)`, `Vh (r, n_cols)` in float64 for the coefficient solve
    (the specification's float64 solve), and the detached expansion point `vbar`. Only the
    rank-`r` factors are kept; with full rank `r = n_cols`. The Gram matrix `J^T J` is never
    formed for the solve: every product with `J`, `J^T` and `J^+` goes through the factors.
    """

    def __init__(self, U: Tensor, S: Tensor, Vh: Tensor, vbar: Tensor, report: RankReport,
                 column_scale: Optional[Tensor] = None):
        self.U, self.S, self.Vh = U, S, Vh
        # If present, the stored SVD is of J diag(1 / column_scale), not J itself.  The public
        # products below still implement the ORIGINAL J.  Keeping None for the unscaled case is
        # deliberate: the established ten-column tangent arm executes its old algebra exactly.
        self.column_scale = column_scale
        self.vbar = vbar.detach().clone()
        self.report = report
        self.n_rows, self.n_cols = report.n_rows, report.n_cols
        self.rank = report.rank

    # ------------------------------------------------------------------ construction
    @staticmethod
    def rank_tolerance(S: Tensor, n_rows: int, n_cols: int, rtol: Optional[float] = None) -> float:
        # THEORY: numpy.linalg.matrix_rank default, S.max() * max(M, N) * eps(dtype)
        # (Golub and Van Loan, Matrix Computations, 4th ed., sect. 5.4.1 numerical rank).
        if rtol is None:
            rtol = max(n_rows, n_cols) * torch.finfo(S.dtype).eps
        return float(S[0]) * float(rtol)

    @classmethod
    def from_matrix(cls, J: Tensor, vbar: Tensor, rtol: Optional[float] = None,
                    build_seconds: float = 0.0, scale_columns: bool = False) -> 'OBCBasis':
        """Factorise a float64 `J (n_rows, n_cols)`. `no_grad` throughout."""
        if J.dtype != torch.float64:
            raise TypeError('the reference basis must be built in float64, got %s' % J.dtype)
        if J.dim() != 2 or min(J.shape) == 0:
            raise ValueError('the reference basis must be a non-empty matrix, got %s'
                             % (tuple(J.shape),))
        with torch.no_grad():
            n_rows, n_cols = J.shape
            # The affine offset column has different physical units and can be orders of
            # magnitude larger than a parameter sensitivity.  Equilibrating columns makes the
            # rank decision and pseudoinverse invariant to those arbitrary column magnitudes.
            # Zero columns retain scale one so rank loss is reported by the normal policy.
            G_raw = J.T @ J
            norms_raw = G_raw.diagonal().clamp_min(0).sqrt()
            column_scale = None
            J_factor = J
            if scale_columns:
                column_scale = torch.where(norms_raw > 0, norms_raw, torch.ones_like(norms_raw))
                J_factor = J / column_scale
            t0 = time.perf_counter()
            U, S, Vh = torch.linalg.svd(J_factor, full_matrices=False)
            svd_seconds = time.perf_counter() - t0
            declared_rtol = (max(n_rows, n_cols) * torch.finfo(S.dtype).eps
                             if rtol is None else float(rtol))
            tol = cls.rank_tolerance(S, n_rows, n_cols, rtol)
            rank = int((S > tol).sum())
            cond = float(S[0] / S[rank - 1]) if rank > 0 else float('inf')
            # Report correlations and norms in the ORIGINAL (unscaled) coordinates.
            corr = G_raw / (norms_raw[:, None] * norms_raw[None, :]).clamp_min(1e-300)
            report = RankReport(
                n_rows=n_rows, n_cols=n_cols,
                singular_values=[float(s) for s in S],
                # Store the DECLARED relative tolerance. `tol / S[0]` is undefined for the
                # legitimate rank-zero case J=0, which the lifecycle must report and reject via
                # its normal rank-loss policy rather than failing here with a division by zero.
                rank=rank, tol=tol, rtol=float(declared_rtol), cond=cond,
                col_norms=[float(v) for v in norms_raw], col_corr=corr.tolist(),
                build_seconds=build_seconds, svd_seconds=svd_seconds,
                extra={'column_scaled': bool(scale_columns),
                       'column_scale': None if column_scale is None
                       else [float(v) for v in column_scale]})
            if rank < n_cols:
                # Keep only the numerically nonzero directions in the SOLVE factors. The projector
                # onto range(J) is then the rank-r one; this is what the Moore-Penrose inverse
                # does, and it is reported, never silent (the lifecycle decides what to do).
                U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
        return cls(U, S, Vh, vbar, report, column_scale=column_scale)

    # ------------------------------------------------------------------ products
    def coefficient(self, F: Tensor) -> Tensor:
        """`theta = J^+ F = V S^-1 U^T F`, float64, differentiable in `F` (any float dtype)."""
        if F.dim() != 1 or F.shape[0] != self.n_rows:
            raise ValueError('field must be a flat (%d,) stack, got %s' % (self.n_rows, tuple(F.shape)))
        F64 = F.to(torch.float64)
        theta = self.Vh.T @ ((self.U.T @ F64) / self.S)
        return theta if self.column_scale is None else theta / self.column_scale

    def apply(self, theta: Tensor) -> Tensor:
        """`J theta` (float64, n_rows) through the factors."""
        theta64 = theta.to(torch.float64)
        if self.column_scale is not None:
            theta64 = self.column_scale * theta64
        return self.U @ (self.S * (self.Vh @ theta64))

    def jt(self, F: Tensor) -> Tensor:
        """`J^T F` (float64, n_cols) through the factors."""
        out = self.Vh.T @ (self.S * (self.U.T @ F.to(torch.float64)))
        return out if self.column_scale is None else self.column_scale * out

    def project_onto(self, F: Tensor) -> Tensor:
        """`P_J F = U U^T F` (float64)."""
        F64 = F.to(torch.float64)
        return self.U @ (self.U.T @ F64)

    def project_out(self, F: Tensor) -> Tensor:
        """`F - J theta(F)` in float64, on the reference stack (the stacked construction)."""
        return F.to(torch.float64) - self.apply(self.coefficient(F))

    # ------------------------------------------------------------------ diagnostics
    def r_perp(self, F_tilde: Tensor, eps: float = 1e-300) -> float:
        """`||J^T F_tilde|| / (||J||_F ||F_tilde|| + eps)`, the scale-free orthogonality residual."""
        with torch.no_grad():
            Ft = F_tilde.to(torch.float64)
            num = torch.linalg.vector_norm(self.jt(Ft))
            den = self.frobenius() * torch.linalg.vector_norm(Ft) + eps
            return float(num / den)

    def r_overlap(self, F: Tensor, eps: float = 1e-300) -> float:
        """`rho(J; F) = ||J J^+ F|| / (||F|| + eps)`: the in-span fraction of a field."""
        with torch.no_grad():
            F64 = F.to(torch.float64)
            return float(torch.linalg.vector_norm(self.project_onto(F64))
                         / (torch.linalg.vector_norm(F64) + eps))

    rho = r_overlap

    def frobenius(self) -> float:
        if self.column_scale is None:
            return float(torch.linalg.vector_norm(self.S))
        raw_right = self.Vh * self.column_scale[None, :]
        return float(torch.linalg.vector_norm(self.S[:, None] * raw_right))

    def relative_change(self, other: 'OBCBasis') -> float:
        """`||J - J_other||_F / ||J_other||_F` through the factors (same rows and columns)."""
        with torch.no_grad():
            if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
                raise ValueError('relative basis change needs equal shapes, got %s and %s'
                                 % ((self.n_rows, self.n_cols),
                                    (other.n_rows, other.n_cols)))
            # Do not use ||A||^2 + ||B||^2 - 2 tr(A^T B): successive epoch bases can be nearly
            # equal, and that expression catastrophically cancels exactly where r_J_step is
            # meant to be informative. Reconstruct only small row chunks from the low-rank
            # factors and accumulate the difference norm; no full J is materialized.
            dev = self.U.device
            oU, oS, oVh = other.U.to(dev), other.S.to(dev), other.Vh.to(dev)
            right = self.Vh if self.column_scale is None else (
                self.Vh * self.column_scale[None, :])
            oright = oVh if other.column_scale is None else (
                oVh * other.column_scale.to(dev)[None, :])
            diff2 = torch.zeros((), dtype=torch.float64, device=dev)
            for start in range(0, self.n_rows, 65536):
                stop = min(start + 65536, self.n_rows)
                A = (self.U[start:stop] * self.S) @ right
                B = (oU[start:stop] * oS) @ oright
                diff2 = diff2 + ((A - B) ** 2).sum()
            denominator = other.frobenius()
            return float(diff2.sqrt()) / denominator if denominator else float('inf')

    # ------------------------------------------------------------------ housekeeping
    def to(self, device) -> 'OBCBasis':
        self.U, self.S, self.Vh = self.U.to(device), self.S.to(device), self.Vh.to(device)
        self.vbar = self.vbar.to(device)
        if self.column_scale is not None:
            self.column_scale = self.column_scale.to(device)
        return self

    @property
    def device(self):
        return self.U.device

    def memory_bytes(self) -> int:
        tensors = (self.U, self.S, self.Vh, self.vbar)
        if self.column_scale is not None:
            tensors += (self.column_scale,)
        return sum(t.numel() * t.element_size() for t in tensors)


# ---------------------------------------------------------------------- free functions
def build_stacked_sensitivity(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor, Z: Tensor,
                              row_ix: Sequence[int], chunk: int = 16384,
                              device=None) -> Tensor:
    """`J (N * n_rows_sel, p)` in float64: `d step(v, z_k) / dv` at `vbar`, stacked sample-major.

    `step(v, Zc)` maps `(p,)` and `(Nc, nz, 1)` to `(Nc, nx, 1)`; `row_ix` selects the rows of
    `x_plus` that are protected (the rows the learning component writes). Forward-mode
    `jacfwd` over the BATCHED step, chunked so the tangent-carrying intermediates fit the
    device (stage 0 of D-190 measured this form at half the cost of `vmap(jacfwd(single))`).
    Everything is detached: no gradient passes through the basis construction.
    """
    if vbar.dtype != torch.float64:
        raise TypeError('build in float64: vbar is %s' % vbar.dtype)
    dev = device if device is not None else vbar.device
    row_ix = torch.as_tensor(list(row_ix), dtype=torch.long, device=dev)
    vbar = vbar.detach().to(dev)
    out = []
    with torch.no_grad():
        for s in range(0, Z.shape[0], chunk):
            Zc = Z[s:s + chunk].to(device=dev, dtype=torch.float64)
            Jc = torch.func.jacfwd(lambda v: step(v, Zc))(vbar)      # (Nc, nx, 1, p)
            Jc = Jc[:, row_ix, 0, :]                                   # (Nc, nr, p)
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


def directional_correction(step: Callable[[Tensor, Tensor], Tensor], vbar: Tensor,
                           theta: Tensor, Z: Tensor) -> Tensor:
    """`J(vbar, z) theta` as ONE forward-mode JVP of `step` in direction `theta`: `(N, nx, 1)`.

    `theta` may carry autograd history (it is a function of the learning component); the
    tangent output is then differentiable with respect to it (reverse-over-forward), which is
    what lets gradients flow through the coefficient to the learning component's weights.
    """
    return torch.func.jvp(lambda v: step(v, Z), (vbar,), (theta,))[1]


# ---------------------------------------------------------------------- lifecycle
class OBCLifecycle:
    """Once-per-objective coefficient and the epoch-refresh policy.

    Parameters
    ----------
    reference_field : callable(learning_component) -> Tensor (n_rows,)
        Evaluates the CURRENT learning component on the fixed reference set and returns the
        protected rows stacked sample-major, in the pipeline dtype, WITH autograd history.
    basis_builder : callable(vbar_detached: Tensor float64) -> (J float64, build_seconds)
        Builds the stacked sensitivity at a parameter point. Supplied by the system adapter.
    refresh : 'epoch' | 'fixed'
        'epoch' rebuilds at every epoch boundary at the current parameter point; 'fixed'
        builds once at the first objective and keeps that basis (the fixed-initial-basis
        diagnostic arm of the specification).
    out_dtype : torch.dtype
        The pipeline dtype the coefficient is cast to at the rollout boundary.
    """

    def __init__(self, reference_field: Callable, basis_builder: Callable, refresh: str = 'epoch',
                 rtol: Optional[float] = None, out_dtype: torch.dtype = torch.float32,
                 expected_rank: Optional[int] = None, max_consecutive_rank_loss: int = 2,
                 verbose: bool = True, name_columns: Optional[Sequence[str]] = None,
                 scale_columns: bool = False):
        if refresh not in ('epoch', 'fixed'):
            raise ValueError("refresh must be 'epoch' or 'fixed', got %r" % (refresh,))
        self.reference_field = reference_field
        self.basis_builder = basis_builder
        self.refresh_policy = refresh
        self.rtol = rtol
        self.out_dtype = out_dtype
        self.expected_rank = expected_rank
        self.max_consecutive_rank_loss = int(max_consecutive_rank_loss)
        self.verbose = verbose
        self.names = list(name_columns) if name_columns is not None else None
        self.scale_columns = bool(scale_columns)

        self.basis: Optional[OBCBasis] = None
        self.basis0: Optional[OBCBasis] = None          # the first valid basis, for r_J
        self.last_refresh_epoch: Optional[int] = None
        self.n_refresh = 0
        self.n_rank_loss = 0
        self.consecutive_rank_loss = 0
        self.n_objective_evals = 0
        self.last_theta: Optional[Tensor] = None       # detached float64 copy
        self.last_field_norm: float = float('nan')
        self.history: list = []                        # dicts, one per refresh
        self.timing = {'field_and_solve_s': [], 'refresh_s': []}

    # ------------------------------------------------------------------ refresh
    def refresh(self, vbar: Tensor, epoch: Optional[int] = None) -> RankReport:
        """Rebuild the basis at a DETACHED copy of `vbar`; apply the rank-loss policy."""
        t0 = time.perf_counter()
        vbar64 = vbar.detach().to(torch.float64).clone()
        J, build_s = self.basis_builder(vbar64)
        cand = OBCBasis.from_matrix(J, vbar64, rtol=self.rtol, build_seconds=build_s,
                                    scale_columns=self.scale_columns)
        del J
        if self.basis is not None and cand.device != self.basis.device:
            cand.to(self.basis.device)       # keep every basis where the field is evaluated
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
            if self.verbose:
                print(msg + 'Retaining the previous basis and its vbar (%d consecutive).'
                      % self.consecutive_rank_loss, flush=True)
                for ln in rep.lines(self.names):
                    print(ln, flush=True)
            entry['retained_previous'] = True
        else:
            self.consecutive_rank_loss = 0
            prev = self.basis
            self.basis = cand
            if self.basis0 is None:
                self.basis0 = cand
            entry['r_J'] = cand.relative_change(self.basis0) if self.basis0 is not cand else 0.0
            entry['r_J_step'] = cand.relative_change(prev) if prev is not None else 0.0
            if self.verbose:
                print('[obc] refresh at epoch %s: vbar=%s' % (
                    epoch, ' '.join('%+.4e' % float(v) for v in vbar64)), flush=True)
                for ln in rep.lines(self.names):
                    print(ln, flush=True)
                print('  r_J = %.3e (vs first basis), r_J_step = %.3e (vs previous)'
                      % (entry['r_J'], entry['r_J_step']), flush=True)
        self.n_refresh += 1
        self.last_refresh_epoch = epoch
        entry['seconds'] = time.perf_counter() - t0
        self.timing['refresh_s'].append(entry['seconds'])
        self.history.append(entry)
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

    # ------------------------------------------------------------------ objective
    def coefficient(self, learning_component) -> Tensor:
        """`theta_aux` for THIS objective evaluation: the current learning component on the
        full reference set, solved in float64 with autograd, cast to the pipeline dtype.

        Called exactly once per objective; never cache the result across optimizer updates
        and never call this inside a rollout timestep.
        """
        if self.basis is None:
            raise RuntimeError('OBC coefficient requested before any basis was built; call '
                               'maybe_refresh/refresh first.')
        t0 = time.perf_counter()
        F = self.reference_field(learning_component)
        if F.device != self.basis.device:
            self.basis.to(F.device)
        theta64 = self.basis.coefficient(F)
        self.n_objective_evals += 1
        self.last_theta = theta64.detach().clone()
        self.last_field_norm = float(torch.linalg.vector_norm(F.detach()))
        self.timing['field_and_solve_s'].append(time.perf_counter() - t0)
        return theta64.to(self.out_dtype)

    def prediction_coefficient(self, learning_component) -> Tensor:
        """Current coefficient for validation/prediction, without training bookkeeping.

        An optimizer step changes the learning component after ``coefficient`` populated the
        model's frozen prediction buffer.  Validation must therefore solve once more from the
        updated component.  This is deliberately a separate, no-grad path: it must not retain an
        objective graph or increment ``n_objective_evals``/its timing diagnostics.
        """
        if self.basis is None:
            raise RuntimeError('OBC prediction coefficient requested before a basis was built.')
        with torch.no_grad():
            F = self.reference_field(learning_component)
            if F.device != self.basis.device:
                self.basis.to(F.device)
            return self.basis.coefficient(F).to(self.out_dtype)

    # ------------------------------------------------------------------ diagnostics
    def stacked_diagnostics(self, learning_component) -> dict:
        """`r_perp` and `r_overlap` of the current learning component on the reference stack,
        using the stored factors (the float64 construction). No grad."""
        with torch.no_grad():
            F = self.reference_field(learning_component).detach()
            if F.device != self.basis.device:
                self.basis.to(F.device)
            theta = self.basis.coefficient(F)
            Ft = F.to(torch.float64) - self.basis.apply(theta)
            return {'r_perp': self.basis.r_perp(Ft), 'r_overlap': self.basis.r_overlap(F),
                    'field_norm': float(torch.linalg.vector_norm(F)),
                    'theta': theta.tolist()}

    def memory_bytes(self) -> int:
        return self.basis.memory_bytes() if self.basis is not None else 0

    def state_summary(self) -> dict:
        return {'n_refresh': self.n_refresh, 'n_rank_loss': self.n_rank_loss,
                'n_objective_evals': self.n_objective_evals,
                'last_refresh_epoch': self.last_refresh_epoch,
                'refresh': self.refresh_policy,
                'rank': None if self.basis is None else self.basis.rank,
                'cond': None if self.basis is None else self.basis.report.cond}
