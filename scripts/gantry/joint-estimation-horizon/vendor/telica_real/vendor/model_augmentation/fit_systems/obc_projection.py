"""Orthogonal-by-construction augmentation at the state level (D-190).

__project_origin__ = "added"

Gyorok's 2026 orthogonal-by-construction parameterization, carried from the input-output setting
of the source to the state equation of the gantry, whose baseline is RATIONAL in its parameters
and therefore has no exact linear-in-parameters regressor.

THE CONSTRUCTION. Let `vartheta` be the ten identifiable combinations and `vartheta_bar` a
declared expansion point. The affine Taylor expansion of the baseline transition at that point,

    f_vartheta(x, u) ~= f_vartheta_bar(x, u) + J(x, u) (vartheta - vartheta_bar),

supplies the protected basis `F(x, u) = [J(x, u) | c(x, u)]`, `c = f_vartheta_bar`. Stack it over
a point set, fit a coefficient from the current learned write, and subtract:

    a      = K_ref  col_i R_p g_eta(z_ref(i))          (K_ref a held pseudo-inverse)
    g~(z)  = g(z) - [J(z) | c(z)] a .

THE TAYLOR APPROXIMATION CONSTRUCTS THE BASIS AND NOTHING ELSE. The prediction keeps the exact
nonlinear baseline transition; the expansion is never substituted into the rollout.

WHAT THE ROLLOUT ACTUALLY EVALUATES. Not `J(z_k)`. With `a` fixed for the pass the correction is
`J(z_k) a[0:10] + c(z_k) a[10]`, whose first term is a single Jacobian-vector product in the
direction `a[0:10]` and whose second is the nominal transition the rollout has already computed.
Measured (`implementation/00-cost/`): the ten-column Jacobian costs 15.2x a plain block step at
the production batch, one directional derivative costs 9.0x, and a training step forward AND
backward costs 3.19x more with the subtraction than without. The ten-column Jacobian is built
only on the construction evaluations, once per epoch, under `no_grad`.

THE GRADIENT. `a` is recomputed on every forward pass and DIFFERENTIATED THROUGH; this is not
optional. Measured on the investigation prototype, `norm(Phi' dG~/deta)` is `1.29e-13` when the
coefficient is differentiated through and `4.49e-02` when it is detached, the latter being exactly
the unprojected value. The pseudo-inverse is prepared once when the basis is built and held.

THE ONE DELIBERATE APPROXIMATION. At each epoch boundary `vartheta_bar`, `J`, `c` and the
factorization are rebuilt under `no_grad` and then held fixed for the epoch, so no gradient
reaches `vartheta` through the basis. It must be stated wherever the method is described.
Expansion-point sensitivity is mild and roughly linear: maximum principal angle to the nominal
span `1.5e-03 rad` at 1 percent parameter displacement, `7.3e-03` at 5 percent, `2.9e-02` at 20.

ADDITIONAL STATES. All `n_p + n_a` rows are stacked. The baseline defines no update equation for
the additional states, so those rows of the basis are exactly zero, `(I - P)` leaves `g_a`
untouched and the coefficient ignores it. That is an assertion the tests check, not a special case
the code handles.

THE OFFSET COLUMN IS A NUMERICAL HAZARD. Its singular value is about 1030x the largest parameter
column, because it is the nominal transition itself with position rows of order 0.3 m against
parameter columns of order 1e-3. A cutoff stated relative to `sigma_0` is therefore a cutoff
against the offset: at `rcond = 1e-4` the rank collapses 11 to 6, silently discarding five genuine
parameter directions. This module SCALES THE COLUMNS to unit norm before the pseudo-inverse, which
makes a `sigma_0`-relative statement legitimate again. Scaling changes the coefficient but not the
span and not the applied correction, since `F D (D^-1 a) = F a`.

WHAT MAY BE CLAIMED. One-step non-overlap of the learned physical write with the affine baseline
tangent set over a FROZEN reference set, at any rank; exact block structure of the joint parameter
Jacobian over that same set, given the coefficient is differentiated through; and removal of the
source's Example 2 flat direction at the level of the one-step augmented map. No consistency, no
asymptotics, nothing at horizon level. No claim may be worded POINTWISE: only the stacked sum
vanishes (`1.29e-13`), the median per-sample product is `1.88e-03`, and the source's Eq. (40)
is wrong to display it per data point. `rho` is the reported metric, never a raw inner product:
the raw metric is dominated by the offset direction and inverted the conclusion once already.

AND THE CONDITION THAT DECIDES WHETHER ANY OF IT HELPS. Gyorok's Condition 4, that the true
unmodeled term is orthogonal to the baseline regressor on the data. Measured on the same rig the
SIGN of the effect flips with it: at `rho(Phi; delta) = 2.7e-15` the projected arm gives
combination error 0.0206 against 0.0891 unprojected; at `rho = 0.9465` it gives 0.4175, four times
worse than no training at all. Report an estimate of `rho(Phi; delta)` beside every recovery
result.
"""

__project_origin__ = "added"

from typing import Optional

import torch
from torch import Tensor, nn
from torch.func import functional_call, jacfwd, jvp

from model_augmentation.fit_systems.blocks import Block
from model_augmentation.utils.utils import added

# HEURISTIC: default cutoff on the COLUMN-SCALED spectrum. `1e-10` follows the investigation
# prototype (`investigation-20260915/obc_proj.py` RCOND). The column scaling is what makes a
# sigma_0-relative cutoff a legitimate statement here; without it this number would be a cutoff
# against the offset column, which is the documented failure.
DEFAULT_RCOND = 1e-10


# ---------------------------------------------------------------------------- the baseline maps
def reduced_step(block, free_params: Tensor, z: Tensor) -> Tensor:
    """One physical-block step with the reduced free coordinate supplied functionally.

    `z` is `(B, n_p + n_u, 1)` in the block's own NORMALISED coordinates; the return is
    `(B, n_p, 1)`.
    """
    return functional_call(block, {'free_params': free_params}, (z,))


def stacked_basis(block, Z: Tensor, v_bar: Optional[Tensor] = None, chunk: int = 512):
    """The stacked protected basis `[J | c]` over a point set at a declared expansion point.

    Parameters
    ----------
    block : Reduced_Gantry_State_Block, float64, eval mode.
    Z : (M, n_p + n_u, 1) block inputs at the construction evaluations.
    v_bar : (10,) the expansion point in the reduced FREE coordinate. Defaults to the block's
        CURRENT `free_params`, detached, which is what an epoch-boundary rebuild wants. It must
        then be held and handed to `directional` for the rest of the epoch: the parameters keep
        training, so re-reading them later would silently move the expansion point and the basis
        would no longer be the one the coefficient was fitted against.

    Returns
    -------
    J : (M*n_p, 10)   d f / d free_params at `v_bar`, sample-major
    c : (M*n_p, 1)    f at `v_bar`. In the reduced/log coordinate the offset IS the nominal
                      transition, not `f - J vartheta_bar`; both give the same span (verified to
                      `2.98e-08 rad` maximum principal angle) and only the conditioning of the
                      coefficient moves.
    v_bar : (10,)     the expansion point actually used, detached.

    `jacfwd` over the BATCHED step, not `vmap(jacfwd(f_single))`: the same matrix to `2.1e-16`
    relative at half the cost, because the per-sample composition evaluates the block at batch one
    and loses every BLAS economy the batched step has (`implementation/00-cost/`).
    """
    M = Z.shape[0]
    v_bar = (block.free_params.detach().clone() if v_bar is None
             else v_bar.detach().clone())
    Js, cs = [], []
    with torch.no_grad():
        for s in range(0, M, chunk):
            zc = Z[s:s + chunk]

            def f(v, zc=zc):
                return reduced_step(block, v, zc).reshape(-1)

            Js.append(jacfwd(f)(v_bar))
            cs.append(f(v_bar).unsqueeze(-1))
    block._cur = None
    return torch.cat(Js, dim=0), torch.cat(cs, dim=0), v_bar


def directional(block, z: Tensor, a: Tensor, v_bar: Tensor) -> Tensor:
    """`J(z) a[0:10] + c(z) a[10]`, the correction the rollout applies, as ONE jvp.

    Exact: `jvp(f, v_bar, a[0:10])` is `J @ a[0:10]` to round-off (verified `4.6e-16` relative).
    The primal the same call returns is `c(z)`, so the nominal transition costs nothing extra.

    `v_bar` is the FROZEN expansion point the basis was built at, not the block's live
    `free_params`. Reading the live value here would move the expansion point mid-epoch while the
    coefficient still belonged to the old one.

    `a` may carry gradient; the jvp is differentiable in its tangent, which is what makes the
    coefficient differentiated-through.
    """
    prim, tang = jvp(lambda v: reduced_step(block, v, z), (v_bar,), (a[:-1],))
    return tang + prim * a[-1]


# ------------------------------------------------------------------------------- the rank policy
def rank_report(F: Tensor, rcond: float = DEFAULT_RCOND,
                tols=(1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14)):
    """Spectrum, rank at several tolerances, and the gap, on the COLUMN-SCALED matrix.

    Returns a dict. `gap` is the ratio of the last kept to the first discarded singular value at
    `rcond`; it is `inf` when nothing is discarded.
    """
    D = F.abs().pow(2).sum(dim=0).sqrt().clamp(min=1e-300)
    S = torch.linalg.svdvals(F / D)
    ratios = (S / S[0]).tolist()
    keep = int((S > rcond * S[0]).sum())
    gap = float('inf') if keep == S.numel() else float(S[keep - 1] / S[keep].clamp(min=1e-300))
    return {
        'n_rows': int(F.shape[0]), 'n_cols': int(F.shape[1]),
        'singular_values_scaled': S.tolist(),
        'ratio_to_sigma0': ratios,
        'rank_at_rtol': {f'{t:.0e}': int((S > t * S[0]).sum()) for t in tols},
        'rank_at_rcond': keep,
        'gap_last_kept_to_first_discarded': gap,
        'condition_number': float(S[0] / S[keep - 1].clamp(min=1e-300)),
        'column_norms': D.tolist(),
    }


def principal_angles(A: Tensor, B: Tensor):
    """Principal angles in radians between `range(A)` and `range(B)`, largest first."""
    QA = torch.linalg.qr(A)[0]
    QB = torch.linalg.qr(B)[0]
    s = torch.linalg.svdvals(QA.T @ QB).clamp(-1.0, 1.0)
    return torch.arccos(s).flip(0)


class RankLossAbort(RuntimeError):
    """Raised when the protected basis loses rank and the policy is to abort.

    Silent truncation is forbidden (handoff Sect. 12). Either the rank drop is reported and
    accepted, in which case the construction continues on the reduced span and every discarded
    singular value plus the principal angles to the previous span are logged, or it aborts here.
    """


# ------------------------------------------------------------------------------------ the basis
@added
class OBCBasis:
    """The frozen protected basis of one epoch: `[J | c]`, its pseudo-inverse, and its rank.

    Built under `no_grad` at an epoch boundary and held fixed until the next one. Holds no
    gradient and is not an `nn.Module`: it is state of the training loop, not of the model.
    """

    def __init__(self, J: Tensor, c: Tensor, v_bar: Optional[Tensor] = None,
                 rcond: float = DEFAULT_RCOND,
                 previous: Optional['OBCBasis'] = None, expected_rank: Optional[int] = None,
                 on_rank_loss: str = 'report', log=print):
        self.F = torch.cat([J, c], dim=1)
        # The expansion point this basis belongs to, carried with it so the two cannot drift
        # apart. `directional` must be handed THIS, not the block's live parameters.
        self.v_bar = None if v_bar is None else v_bar.detach().clone()
        self.rcond = float(rcond)
        self.report = rank_report(self.F, rcond)
        self.rank = self.report['rank_at_rcond']

        # Column scaling to unit norm BEFORE the pseudo-inverse. Without it the cutoff is a cutoff
        # against the offset column, whose singular value is about 1030x the largest parameter
        # column. The applied correction is unchanged: `F D (D^-1 a) = F a`.
        self._D = self.F.abs().pow(2).sum(dim=0).sqrt().clamp(min=1e-300)
        Fs = self.F / self._D
        U, S, Vh = torch.linalg.svd(Fs, full_matrices=False)
        keep = S > self.rcond * S[0]
        self._U_kept = U[:, :int(keep.sum())]
        Sinv = torch.where(keep, 1.0 / S, torch.zeros_like(S))
        # `K_scaled @ G` gives the coefficient in the SCALED basis; dividing by `D` returns it to
        # the unscaled one, which is what `directional` consumes.
        self._K = (Vh.T @ torch.diag(Sinv) @ U.T) / self._D.unsqueeze(-1)

        self._check_rank(S, keep, previous, expected_rank, on_rank_loss, log)

    def _check_rank(self, S, keep, previous, expected_rank, on_rank_loss, log):
        """Report, and never silently truncate."""
        n_disc = int((~keep).sum())
        self.rank_loss = False
        if expected_rank is not None and self.rank < expected_rank:
            self.rank_loss = True
        if n_disc:
            disc = S[~keep]
            log(f'[obc] basis rank {self.rank}/{S.numel()} at rcond {self.rcond:.0e}; '
                f'discarded singular values (column-scaled, relative to sigma_0): '
                f'{[float(v / S[0]) for v in disc]}')
        if previous is not None:
            ang = principal_angles(self.range_basis(), previous.range_basis())
            self.angles_to_previous = [float(a) for a in ang]
            log(f'[obc] principal angles to the previous epoch span [rad]: '
                f'max {max(self.angles_to_previous):.3e}, '
                f'min {min(self.angles_to_previous):.3e}')
        else:
            self.angles_to_previous = None
        if self.rank_loss:
            msg = (f'[obc] RANK LOSS: rank {self.rank} below the expected {expected_rank}. '
                   f'The protected set has shrunk, so directions the baseline can represent are '
                   f'no longer being removed from the learned write.')
            if on_rank_loss == 'abort':
                raise RankLossAbort(msg)
            log(msg + ' Policy is "report": continuing on the REDUCED span.')

    # ------------------------------------------------------------------------------- use
    def range_basis(self) -> Tensor:
        """Orthonormal basis of `range(F)`, rank truncated. `(n_rows, rank)`."""
        return self._U_kept

    def coefficient(self, G: Tensor) -> Tensor:
        """`a = K G`. Differentiable in `G`, which is what carries the gradient to the ANN."""
        return self._K @ G

    def rho(self, G: Tensor) -> float:
        """Fraction of `norm(G)` lying in `range(F)`. Scale free, in [0, 1]. THE metric.

        A raw inner product is never reported: it is dominated by the offset direction and
        inverted the conclusion in the investigation, showing a 5x apparent degradation where
        `rho` showed an improvement from 0.4295 to 0.2843.
        """
        Q = self._U_kept
        return float(torch.linalg.vector_norm(Q @ (Q.T @ G))
                     / torch.linalg.vector_norm(G).clamp(min=1e-300))

    def residual_ratio(self, G: Tensor) -> float:
        """`norm(F' G~) / (norm(F) norm(G~))`, the scale-free orthogonality residual."""
        Gt = G - self.F @ self.coefficient(G)
        num = torch.linalg.vector_norm(self.F.T @ Gt)
        den = (torch.linalg.matrix_norm(self.F)
               * torch.linalg.vector_norm(Gt)).clamp(min=1e-300)
        return float(num / den)

    def spectrum(self) -> dict:
        return dict(self.report)


# ----------------------------------------------------------------------------- the wrapper block
@added
class OBC_Projected_ANN_Block(Block):
    """A learned block that subtracts its baseline-representable component, pointwise.

    Wraps a `Static_ANN_Block` and is registered in the `Interconnect` in its place. The
    interconnect calls `forward(z)` once per timestep per batch, so there is no stacked ANN write
    to intercept and no whole-window projector is possible; the subtraction has to be pointwise,
    which is exactly how the source deploys it: the coefficient is one global vector and
    `g~(z) = g(z) - F(z) a` is evaluated at whatever point the rollout is at. No framework
    surgery, no patching, no monkey-patched `optimizer.step`.

    `enabled = False` returns the wrapped block's output tensor ITSELF, so the disabled path is
    bit-identical to the pipeline without this class and not merely numerically equal.

    LIFECYCLE. `rebuild_basis(Z_ref)` at each epoch boundary: it reads the physical block's
    current parameters as the expansion point, builds `[J | c]` over the construction evaluations
    under `no_grad`, factorises once, and holds all of it for the epoch. Inside the epoch every
    forward refits the COEFFICIENT from the current ANN output and differentiates through it.

    COST. Refitting per timestep costs one extra batched ANN evaluation over the `M` construction
    points plus an `11 x (M n_p)` matvec, which is negligible beside the physical step; the
    coefficient is NOT cached across timesteps, so there is no pass-boundary bookkeeping to get
    wrong. The real charge is the directional derivative, measured at 3.19x on a training step
    forward and backward (`implementation/00-cost/`). The graph does hold one ANN-over-`M`
    subgraph per timestep, so memory grows with `nf * M`; `implementation/06-integration/`
    measures it.
    """

    def __init__(self, ann_block, phys_block, route_ix, n_p: int, n_a: int,
                 enabled: bool = False, name: Optional[str] = None) -> None:
        # A `Block`, not a bare `nn.Module`: `Interconnect.determine_signal_ix` resolves a block
        # to its signal index by `isinstance(signal, Block)`, so anything registered in an
        # interconnect has to be one.
        super().__init__(nz=ann_block.nz, nw=ann_block.nw, name=name)
        self.ann = ann_block
        # A PLAIN LIST, so the physical block is NOT registered as a submodule here. It already
        # lives in the interconnect's ModuleList; registering it twice would duplicate its keys in
        # every `state_dict` (`parameters()` dedupes by identity, `state_dict` does not).
        self._phys_ref = [phys_block]
        self.route_ix = tuple(int(i) for i in route_ix)
        self.n_p, self.n_a = int(n_p), int(n_a)
        self.enabled = bool(enabled)

        missing = sorted(set(range(self.n_p)) - set(self.route_ix))
        if missing:
            raise ValueError(
                f'physical rows {missing} are not routed, so the construction cannot subtract on '
                f'them and orthogonality would not hold there. Route every physical row, or do '
                f'not use this wrapper.')
        if len(set(self.route_ix)) != len(self.route_ix):
            raise ValueError('route_ix must not repeat a row')

        self.basis = None
        self._Z_ref = None
        # Column j of the ANN output writes state row route_ix[j]. `_phys_col[r]` is the ANN
        # column writing physical row r; `-1` never occurs because every physical row is routed.
        self.register_buffer(
            '_phys_col',
            torch.tensor([self.route_ix.index(r) for r in range(self.n_p)], dtype=torch.long),
            persistent=False)

    @property
    def phys(self):
        return self._phys_ref[0]

    def init_block(self, z: Tensor):
        return self.ann.init_block(z)

    # -------------------------------------------------------------------------- lifecycle
    def phys_input(self, z: Tensor) -> Tensor:
        """`(B, n_p + n_u, 1)` physical-block input carved out of the ANN input `(B, nz, 1)`."""
        return torch.cat([z[:, :self.n_p, :], z[:, self.n_p + self.n_a:, :]], dim=1)

    def rebuild_basis(self, Z_ref: Tensor, keep_previous: bool = True, **kwargs) -> 'OBCBasis':
        """Rebuild the protected basis at the CURRENT physical parameters. Call at epoch bounds.

        `Z_ref` is `(M, nz, 1)`, the ANN inputs at the construction evaluations, so the physical
        points and the learned-write points are the same points by construction and cannot drift
        apart.
        """
        Zp = self.phys_input(Z_ref)
        J, c, v_bar = stacked_basis(self.phys, Zp)
        prev = self.basis if keep_previous else None
        self.basis = OBCBasis(J, c, v_bar=v_bar, previous=prev, **kwargs)
        self._Z_ref = Z_ref.detach().clone()
        return self.basis

    def clear_basis(self):
        self.basis, self._Z_ref = None, None

    # -------------------------------------------------------------------------- the write
    def _stack_physical_write(self, w: Tensor) -> Tensor:
        """`(M, nw, 1)` ANN output -> `(M*n_p,)` physical-row write, sample-major.

        This is `col_i R_p g(z_i)` of the derivation. The additional-state columns are simply not
        selected, which is the whole of what "the condition sees only the physical rows" means in
        code: no special case, just the routing matrix.
        """
        return w[:, self._phys_col, 0].reshape(-1)

    def coefficient(self) -> Tensor:
        """`a = K col_i R_p g(z_ref(i))` from the CURRENT ANN. Carries gradient by design."""
        return self.basis.coefficient(self._stack_physical_write(self.ann(self._Z_ref)))

    def forward(self, z: Tensor) -> Tensor:
        w = self.ann(z)
        if not self.enabled or self.basis is None:
            return w
        a = self.coefficient()
        delta = directional(self.phys, self.phys_input(z), a, self.basis.v_bar)
        delta = delta.reshape(-1, self.n_p)                     # (B, n_p)
        # Scatter into the ANN's output columns. Columns writing additional-state rows get
        # nothing, because the basis has no rows there; that is asserted in the tests, not
        # special-cased here.
        corr = torch.zeros_like(w[:, :, 0])
        corr[:, self._phys_col] = delta
        return w - corr.unsqueeze(-1)
