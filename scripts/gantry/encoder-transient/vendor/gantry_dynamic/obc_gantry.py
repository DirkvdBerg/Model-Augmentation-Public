"""One-step orthogonal-by-construction (OBC) for the gantry: the GANTRY-SPECIFIC half (D-192).

The generic construction (`model_augmentation/fit_systems/obc.py`) knows a `step(v, z)`, a
reference set and a field. This module supplies everything gantry:

* `GantryReferenceSet`     the fixed one-step reference tuples from every training sample:
                           `q = P^-T y`, `qdot` by a fourth-order central difference inside each
                           record, `x_a = 0`, `u` the recorded `u_total` after the pipeline's
                           block-mean resampling; all in the block's normalized coordinates.
* `GantryOBCCorrection`    the nn.Module attached as `Interconnect.obc_correction`: subtracts
                           `E_route J_b(vbar, x_b, u) theta` (or the affine extension
                           `E_route[J_b theta + alpha f_b(vbar,x_b,u)]`) through
                           ONE forward-mode JVP of `Reduced_Gantry_State_Block.transition_from_free`,
                           with exact zeros on the additional-state rows. Holds `vbar` and the
                           frozen prediction coefficient as buffers.
* `attach_obc`             wires reference set, correction and `OBCLifecycle` onto a built
                           `fit_sys` (called by `model.build_model` when `cfg.obc`).

The controller never enters: the reference tuples are independent one-step evaluations at the
recorded plant input, not a rollout.
"""
__project_origin__ = "added"

import copy
import os
import time
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from scipy.io import loadmat

from model_augmentation.systems.gantry_ss import P
from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block, Static_ANN_Block
from model_augmentation.fit_systems.obc import (
    OBCLifecycle, StaticPassBuffers, build_stacked_sensitivity, directional_correction,
    evaluate_step_rows)

from .config import RunConfig
from .data import TRAIN_FILES, traj_dir, _resample_u, _load_u

COMBO_NAMES = Reduced_Gantry_State_Block.COMBO_NAMES


# ---------------------------------------------------------------------- reference set
@dataclass
class GantryReferenceSet:
    """Fixed one-step reference tuples over all training records (D-192, spec sect. 4)."""
    Z_step: Tensor        # (N, nx_phys + nu, 1) normalized block input, pipeline dtype
    Z_field: Tensor       # (N, nx_phys + nx_ann + nu, 1) normalized ANN input (x_a = 0)
    record_ix: Tensor     # (N,) long, which TRAIN_FILES record each tuple came from
    sample_ix: Tensor     # (N,) long, sample index inside its record (after resampling)
    x_phys: Tensor        # (N, 6) float64, PHYSICAL units [q; qdot] (q logical, m / m/s)
    u_stage: Tensor       # (N, 3) float64, PHYSICAL stage forces [N]
    x_phys_next: Tensor   # (N, 6) float64, the RECORDED next logical state (diagnostics only)
    x_abs: Tensor         # (N, 2) float64, recorded [delta_a, vdelta_a] (diagnostics only)
    n_per_record: list    # samples kept per record
    record_names: list
    ts: float

    @property
    def n(self) -> int:
        return int(self.Z_step.shape[0])

    def memory_bytes(self, hot_only: bool = True) -> int:
        ts = [self.Z_step, self.Z_field]
        if not hot_only:
            ts += [self.record_ix, self.sample_ix, self.x_phys, self.u_stage,
                   self.x_phys_next, self.x_abs]
        return sum(t.numel() * t.element_size() for t in ts)


def fourth_order_central_difference(q: np.ndarray, ts: float) -> np.ndarray:
    """`qdot[k] = (-q[k+2] + 8 q[k+1] - 8 q[k-1] + q[k-2]) / (12 ts)` for k in [2, N-3].

    # THEORY: fourth-order central difference, Fornberg (1988) Math. Comp. 51, Table 1 (first
    # derivative, order 4, stencil -2..2 weights 1/12, -2/3, 0, 2/3, -1/12); the specification's
    # Eq. (reference-reconstruction). Applied only where the full stencil exists: the caller
    # trims two samples from each end of each record and never crosses a record boundary.
    Returns (N-4, ...) aligned with q[2:-2].
    """
    return (-q[4:] + 8.0 * q[3:-1] - 8.0 * q[1:-3] + q[:-4]) / (12.0 * ts)


def _load_record_float64(filename: str, cfg: RunConfig):
    """One training record in float64 with the pipeline's exact resampling.

    `u`: block mean over each hold interval (data.py::_resample_u, D-087). `y`, `x_logical`,
    `delta_a`, `vdelta_a`: point-sampled `[::D]` (data.py::load_traj / load_mat_aug). The
    pipeline casts to float32 for training; the reference derivative is formed from the float64
    source so the fourth-order stencil is not applied to float32-quantized positions.
    """
    d = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    D = cfg.d
    u = _resample_u(_load_u(d), cfg).astype(np.float64)
    N = len(u)
    y = np.asarray(d['y'], dtype=np.float64)[::D][:N]
    x_logical = np.asarray(d['x_logical'], dtype=np.float64)[::D][:N]
    delta_a = np.asarray(d['delta_a'], dtype=np.float64)[::D][:N]
    vdelta_a = np.asarray(d['vdelta_a'], dtype=np.float64)[::D][:N]
    return u, y, x_logical, np.stack([delta_a, vdelta_a], 1)


def build_reference_set(cfg: RunConfig, norm, nx_ann: int, files: Sequence[str] = TRAIN_FILES,
                        stride: int = 1, verbose: bool = True) -> GantryReferenceSet:
    """The fixed reference tuples of D-192 over `files` (default: all fourteen TRAIN_FILES).

    `stride` MUST be 1 for the primary implementation (every valid sample); any other value is a
    measured approximation for probes and tests only and is recorded in the returned set's
    `record_names` tag.
    """
    if cfg.snr is not None:
        raise NotImplementedError(
            'reference construction under output noise (cfg.snr=%r) is not qualified: the '
            'fourth-order derivative of noisy positions would dominate the velocity rows. Build '
            'the reference tuples from a noiseless source or a filtered estimate first.' % cfg.snr)
    ts = cfg.ts_new
    P64 = P.detach().cpu().numpy().astype(np.float64)
    PinvT = np.linalg.inv(P64.T)
    x_mean = np.asarray(norm.x_mean, dtype=np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, dtype=np.float64).reshape(1, 6)
    u_mean = np.asarray(norm.u_mean, dtype=np.float64).reshape(1, 3)
    std_u = np.asarray(norm.std_u, dtype=np.float64).reshape(1, 3)

    xs, us, xn, xa, rec, samp, counts = [], [], [], [], [], [], []
    t0 = time.perf_counter()
    for r, f in enumerate(files):
        u, y, x_log, x_abs = _load_record_float64(f, cfg)
        q = y @ PinvT.T                              # q_k = P^-T y_k, (N, 3)
        qd = fourth_order_central_difference(q, ts)  # (N-4, 3), aligned with q[2:-2]
        k = np.arange(2, len(u) - 2)                 # full stencil exists here only
        if stride > 1:
            k = k[::stride]
            qd = qd[::stride]
        x_ref = np.hstack([q[k], qd])                # (n, 6) physical
        xs.append(x_ref); us.append(u[k])
        xn.append(x_log[k + 1]); xa.append(x_abs[k])
        rec.append(np.full(len(k), r)); samp.append(k); counts.append(len(k))
    x_phys = np.concatenate(xs); u_stage = np.concatenate(us)
    x_next = np.concatenate(xn); x_abs = np.concatenate(xa)
    record_ix = np.concatenate(rec); sample_ix = np.concatenate(samp)

    x_n = (x_phys - x_mean) / std_x
    u_n = (u_stage - u_mean) / std_u
    N = x_n.shape[0]
    dt = cfg.dtype_pt
    Z_step = torch.from_numpy(np.hstack([x_n, u_n])).to(dt).unsqueeze(-1)
    Z_field = torch.from_numpy(
        np.hstack([x_n, np.zeros((N, nx_ann)), u_n])).to(dt).unsqueeze(-1)
    ref = GantryReferenceSet(
        Z_step=Z_step, Z_field=Z_field,
        record_ix=torch.from_numpy(record_ix).long(), sample_ix=torch.from_numpy(sample_ix).long(),
        x_phys=torch.from_numpy(x_phys), u_stage=torch.from_numpy(u_stage),
        x_phys_next=torch.from_numpy(x_next), x_abs=torch.from_numpy(x_abs),
        n_per_record=counts, record_names=[os.path.basename(f) for f in files]
        + ([] if stride == 1 else ['(stride %d: probe only)' % stride]), ts=ts)
    if verbose:
        Y = x_phys[:, 2]
        print('[obc] reference set: %d tuples from %d records (%s per record), stride %d, '
              '%.1f s to build' % (N, len(files), '/'.join(str(c) for c in counts), stride,
                                   time.perf_counter() - t0))
        print('[obc]   Y coverage: min %+.4f max %+.4f mean %+.4f std %.4f m; '
              'hot tensors %.1f MB' % (Y.min(), Y.max(), Y.mean(), Y.std(),
                                       ref.memory_bytes() / 1e6))
    return ref


# ---------------------------------------------------------------------- correction module
class GantryOBCCorrection(nn.Module):
    """`E_route J_b(vbar, x_b, u) theta` for the progressed state, as an nn.Module.

    `forward(x, u, obc_pass)` returns `(nb, nx, 1)`: the correction on the routed PHYSICAL rows,
    exact zeros elsewhere (the additional-state rows are never touched: the expansion matrix
    has zero rows there, and `xp - 0.0` is bit-exact). `obc_pass=None` applies `theta_frozen`,
    the detached coefficient synchronized from the current ANN before validation/prediction
    (and also updated by each objective), which is what diagnostic rollouts use.

    The physical block is held in a plain list so its parameters are not registered twice
    (`parameters()` dedupes by identity, `state_dict()` does not). `vbar`, `theta_frozen` and `E`
    are buffers, so they follow `.to()` / `.cuda()` and are checkpointed with the model.
    """

    def __init__(self, phy_block: Reduced_Gantry_State_Block, nx: int, nx_phys: int, nu: int,
                 route_ix: Sequence[int], dtype: torch.dtype, use_jvp: bool = False,
                 space: str = 'tangent'):
        super().__init__()
        if space not in ('tangent', 'affine'):
            raise ValueError("OBC space must be 'tangent' or 'affine', got %r" % (space,))
        self.space = space
        self._phy = [phy_block]
        # CHANGED (D-194): which implementation of the directional derivative runs.
        # False (the default) is the hand-written forward sensitivity, which compiles.
        # True is `torch.func.jvp`, kept as the reference and used by the equivalence tests.
        self.use_jvp = bool(use_jvp)
        # D-199: set by `attach_obc` from cfg.static_pass_buffers. None = thread fresh tensors.
        self._static = None
        self.nx, self.nx_phys, self.nu = int(nx), int(nx_phys), int(nu)
        route_ix = [int(i) for i in route_ix]
        # Physical rows the ANN writes, and the ANN OUTPUT columns that write them, in the same
        # order, so the field stacking matches the basis stacking row for row.
        self.row_ix = tuple(i for i in route_ix if i < self.nx_phys)
        self.field_cols = tuple(route_ix.index(i) for i in self.row_ix)
        if not self.row_ix:
            raise ValueError('the ANN writes no physical row (route_ix=%r); there is nothing to '
                             'orthogonalize' % (route_ix,))
        p = int(phy_block.free_params.numel())
        E = torch.zeros(self.nx, self.nx_phys, dtype=dtype)
        for i in self.row_ix:
            E[i, i] = 1.0
        self.register_buffer('E', E)
        self.register_buffer('vbar', torch.zeros(p, dtype=dtype))
        self.register_buffer('theta_frozen', torch.zeros(
            p + (1 if self.space == 'affine' else 0), dtype=dtype))

    # -- the generic step(v, z) --------------------------------------------------------
    def step(self, v: Tensor, z: Tensor) -> Tensor:
        return self._phy[0].transition_from_free(v, z)

    def expansion_point(self) -> Tensor:
        """The CURRENT free coordinate (attached); the lifecycle detaches it."""
        return self._phy[0].free_params

    @torch.no_grad()
    def set_frozen(self, theta: Tensor, vbar: Tensor) -> None:
        self.theta_frozen.copy_(theta.detach().to(self.theta_frozen.dtype))
        self.vbar.copy_(vbar.detach().to(self.vbar.dtype))

    # CHANGED (D-194): the ONCE-PER-OBJECTIVE half. `mats_and_tangent_from_free` returns the
    # matrix tuple at the frozen expansion point AND its derivative in the coefficient direction
    # from a single call. Both are state-independent, so computing them once removes two things
    # from the per-timestep path: the functorch transform, and the M(Y) rebuild, which
    # test_tangent.py measures at 2.28x a plain step on its own.
    #
    # THE RESULT IS RETURNED, NOT CACHED ON SELF, and that is the whole point. An earlier version
    # stored it and had `forward` pick the cache when the coefficient matched by identity. Eager
    # that worked. Under Dynamo it silently did not: the identity test compares a graph INPUT
    # against a closed-over constant, is False at trace time, and the compiled graph therefore
    # baked in the fallback branch, running `torch.func.jvp` on EVERY timestep inside inductor.
    # Job 83947 paid 35 s per objective at nf 20 for that. Threading the pair as an argument
    # leaves nothing for the tracer to decide.
    #
    # `theta` carries autograd history back to the ANN, so `dmats` does too. It must be rebuilt
    # on every objective and never reused across optimizer updates (D-192).
    def tangent_pair(self, theta: Tensor):
        """Per-objective physical pass at ``vbar``.

        Tangent OBC returns ``(mats, dmats)``.  Affine OBC returns
        ``(mats, dmats, alpha)`` for ``J a + alpha f(vbar,z)``.  The offset uses the same
        frozen expansion point and the same RK4 primal already produced beside the tangent.
        """
        # Keep the established tangent path's exact tensor, rather than introducing even a
        # harmless view; the affine arm alone splits off its eleventh coefficient.
        direction = theta if self.space == 'tangent' else theta[:-1]
        mats, dmats = self._phy[0].mats_and_tangent_from_free(self.vbar, direction)
        if self._static is not None:
            # D-199: optional fixed addresses, gradients preserved. See StaticPassBuffers.
            mats, dmats = self._static[0](mats), self._static[1](dmats)
        if self.space == 'tangent':
            return mats, dmats
        alpha = theta[-1]
        if self._static is not None:
            alpha = self._static[2]((alpha,))[0]
        return mats, dmats, alpha

    def correction_from_theta(self, x: Tensor, u: Tensor, theta: Tensor,
                              use_jvp: bool = None) -> Tensor:
        """The correction for an explicit coefficient. The equivalence tests use this."""
        if use_jvp if use_jvp is not None else self.use_jvp:
            z = torch.cat([x[:, : self.nx_phys, :], u], dim=1)
            if self.space == 'tangent':
                corr = directional_correction(self.step, self.vbar, theta, z)
            else:
                # One transformed evaluation returns both terms of J a + alpha f(vbar,z).
                fbar, jv = torch.func.jvp(
                    lambda v: self.step(v, z), (self.vbar,), (theta[:-1],))
                corr = jv + theta[-1] * fbar
            return torch.matmul(self.E, corr)
        return self.forward(x, u, self.tangent_pair(theta))

    def forward(self, x: Tensor, u: Tensor, obc_pass=None) -> Tensor:
        """`obc_pass` is the `(mats, dmats)` pair for THIS objective, threaded from `loss`.

        None means no pair was supplied, which is the prediction path (validation, free runs,
        diagnostics). It then builds the pair from `theta_frozen`; the fit system synchronizes
        that buffer from the current ANN before validation and after restoring the final selected
        checkpoint. That path is eager by construction:
        `ClosedLoopSimulator` only routes to the compiled pair under `torch.is_grad_enabled()`.
        """
        xb, ub = x[:, : self.nx_phys, :], u
        if obc_pass is None:
            if self.use_jvp:
                return self.correction_from_theta(
                    x, u, self.theta_frozen, use_jvp=True)
            obc_pass = self.tangent_pair(self.theta_frozen)
        if self.space == 'tangent':
            mats, dmats = obc_pass
        else:
            mats, dmats = obc_pass[:2]
        fbar, jv = self._phy[0]._rk4_with_tangent(
            xb, torch.zeros_like(xb), ub, mats, dmats)
        corr = jv if self.space == 'tangent' else jv + obc_pass[2] * fbar
        return torch.matmul(self.E, corr)                            # (nb, nx, 1)


# ---------------------------------------------------------------------- lifecycle wiring
def make_float64_block(phy_block: Reduced_Gantry_State_Block) -> Reduced_Gantry_State_Block:
    """A float64 copy for the basis build. The live block stays at the pipeline dtype."""
    blk = copy.deepcopy(phy_block)
    blk._cur = None
    return blk.to(torch.float64)


def make_basis_builder(blk64: Reduced_Gantry_State_Block, ref: GantryReferenceSet,
                       row_ix: Sequence[int], chunk: int, device=None,
                       space: str = 'tangent'):
    """`vbar64 -> (J, seconds)` for `OBCLifecycle`: float64 jacfwd over the batched step."""
    def build(vbar64: Tensor):
        # Build where the live parameters are (the training device once fit() has moved the
        # model), unless a device is pinned; the float64 block copy follows.
        dev = device if device is not None else vbar64.device
        if next(blk64.buffers()).device != dev:
            blk64.to(dev)
        t0 = time.perf_counter()
        J = build_stacked_sensitivity(lambda v, z: blk64.transition_from_free(v, z),
                                      vbar64.to(dev), ref.Z_step, row_ix, chunk=chunk, device=dev)
        if space == 'affine':
            # Frozen reference transition, not the live rollout transition.  This is the exact
            # column whose coefficient is fitted on the reference stack.
            c = evaluate_step_rows(lambda v, z: blk64.transition_from_free(v, z),
                                   vbar64.to(dev), ref.Z_step, row_ix,
                                   chunk=chunk, device=dev)
            J = torch.cat((J, c[:, None]), dim=1)
        return J, time.perf_counter() - t0
    build.blk64 = blk64          # exposed for the preflight diagnostics (float64 baseline step)
    return build


def make_reference_field(ref: GantryReferenceSet, field_cols: Sequence[int]):
    """`ann_block -> F (N * n_rows,)`: the ANN on the reference set, routed physical columns,
    stacked sample-major, pipeline dtype, WITH autograd."""
    cols = torch.as_tensor(list(field_cols), dtype=torch.long)
    cache = {}

    def field(ann: Static_ANN_Block) -> Tensor:
        param = next(ann.parameters())
        key = (param.device, param.dtype)
        if cache.get('key') != key:
            cache['Z'] = ref.Z_field.to(device=param.device, dtype=param.dtype)
            cache['cols'] = cols.to(param.device)
            cache['key'] = key
        w = ann(cache['Z'])                                  # (N, nw, 1)
        return w[:, cache['cols'], 0].reshape(-1)             # (N * nr,)
    return field


def attach_obc(fit_sys, cfg: RunConfig, data, norm, ref: Optional[GantryReferenceSet] = None,
               ref_stride: int = 1, verbose: bool = True):
    """Attach reference set, correction module and lifecycle to a built `fit_sys`.

    `ref_stride` is for probes and tests only; the primary implementation uses every sample.
    Returns (lifecycle, correction, reference_set).
    """
    blocks = list(fit_sys.hfn.connected_blocks)
    phy = next((b for b in blocks if isinstance(b, Reduced_Gantry_State_Block)), None)
    ann = next((b for b in blocks if isinstance(b, Static_ANN_Block)), None)
    if phy is None or ann is None:
        raise RuntimeError('OBC needs a Reduced_Gantry_State_Block and a Static_ANN_Block in the '
                           'interconnect (joint_estimation=True, physics_parameterization=reduced).')
    nx = fit_sys.hfn.nx
    if ref is None:
        ref = build_reference_set(cfg, norm, nx_ann=nx - cfg.nx_phys, stride=ref_stride,
                                  verbose=verbose)
    corr = GantryOBCCorrection(phy, nx=nx, nx_phys=cfg.nx_phys, nu=cfg.nu,
                               route_ix=cfg.ann_route_ix, dtype=cfg.dtype_pt,
                               space=cfg.obc_space)
    corr = corr.to(next(phy.parameters()).device)
    if cfg.static_pass_buffers:
        # D-199: optional fixed addresses for the two threaded tuples; measured performance-neutral.
        corr._static = (StaticPassBuffers('mats'), StaticPassBuffers('dmats'))
        if cfg.obc_space == 'affine':
            corr._static += (StaticPassBuffers('alpha'),)
    blk64 = make_float64_block(phy)
    life = OBCLifecycle(
        reference_field=make_reference_field(ref, corr.field_cols),
        basis_builder=make_basis_builder(blk64, ref, corr.row_ix, chunk=cfg.obc_ref_chunk,
                                         space=cfg.obc_space),
        refresh=cfg.obc_refresh, rtol=cfg.obc_rank_rtol, out_dtype=cfg.dtype_pt,
        expected_rank=int(phy.free_params.numel()) + (cfg.obc_space == 'affine'),
        verbose=verbose,
        name_columns=COMBO_NAMES + (['f_bar'] if cfg.obc_space == 'affine' else []),
        scale_columns=(cfg.obc_space == 'affine'))
    fit_sys.hfn.obc_correction = corr
    fit_sys.obc = life
    fit_sys.obc_reference = ref        # kept for diagnostics; not checkpointed (see __getstate__)
    if verbose:
        print('[obc] attached: protected rows %s (ANN cols %s), %d reference tuples, '
              'space=%s, refresh=%s, solve float64, rollout %s'
              % (corr.row_ix, corr.field_cols, ref.n, cfg.obc_space, cfg.obc_refresh,
                 cfg.dtype_pt))
    return life, corr, ref
