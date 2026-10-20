"""One-step orthogonal-by-construction (OBC) for the gantry: the GANTRY-SPECIFIC half (D-192)."""
__project_origin__ = "added"

import copy
import os
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from scipy.io import loadmat

from model_augmentation.systems.gantry_ss import P
from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block, Static_ANN_Block
from model_augmentation.fit_systems.obc import (
    OBCLifecycle, build_stacked_sensitivity,
    evaluate_step_rows)

from .config import RunConfig
from .data import traj_dir, _record_at_fs_new


@dataclass
class GantryReferenceSet:
    """Fixed one-step reference tuples over all training records (D-192, spec sect. 4)."""
    Z_step: Tensor
    Z_field: Tensor
    record_ix: Tensor
    sample_ix: Tensor
    record_names: list
    x_phys_next_rec: Optional[Tensor] = None

    @property
    def n(self) -> int:
        return int(self.Z_step.shape[0])


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
    """One training record in float64 with the pipeline's exact resampling."""
    d = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    u, y = (np.asarray(a, dtype=np.float64) for a in _record_at_fs_new(d, cfg))
    return u, y


def build_reference_set(cfg: RunConfig, norm, nx_ann: int, files: Sequence[str],
                        verbose: bool = True) -> GantryReferenceSet:
    """The fixed reference tuples of D-192 over `files` (the training records)."""
    ts = cfg.ts_new
    P64 = P.detach().cpu().numpy().astype(np.float64)
    PinvT = np.linalg.inv(P64.T)
    x_mean = np.asarray(norm.x_mean, dtype=np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, dtype=np.float64).reshape(1, 6)
    u_mean = np.asarray(norm.u_mean, dtype=np.float64).reshape(1, 3)
    std_u = np.asarray(norm.std_u, dtype=np.float64).reshape(1, 3)

    xs, us, rec, samp, counts, xnr = [], [], [], [], [], []
    t0 = time.perf_counter()
    for r, f in enumerate(files):
        u, y = _load_record_float64(f, cfg)
        q = y @ PinvT.T
        qd = fourth_order_central_difference(q, ts)
        k = np.arange(2, len(u) - (3 if cfg.obc_ref_successor else 2))
        x_ref = np.hstack([q[k], qd[k - 2]])
        if cfg.obc_ref_successor:
            xnr.append(np.hstack([q[k + 1], qd[k - 1]]))
        xs.append(x_ref); us.append(u[k])
        rec.append(np.full(len(k), r)); samp.append(k); counts.append(len(k))
    x_phys = np.concatenate(xs); u_stage = np.concatenate(us)
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
        record_names=[os.path.basename(f) for f in files],
        x_phys_next_rec=torch.from_numpy(np.concatenate(xnr)) if xnr else None)
    if verbose:
        print('[obc] reference set: %d tuples from %d records (%s per record), '
              '%.1f s to build' % (N, len(files), '/'.join(str(c) for c in counts),
                                   time.perf_counter() - t0))
    return ref


class GantryOBCCorrection(nn.Module):
    """`E_route J_b(vbar, x_b, u) theta` for the progressed state, as an nn.Module."""

    def __init__(self, phy_block: Reduced_Gantry_State_Block, nx: int, nx_phys: int, nu: int,
                 route_ix: Sequence[int], dtype: torch.dtype,
                 space: str):
        super().__init__()
        self.space = space
        self._phy = [phy_block]
        self.nx, self.nx_phys, self.nu = int(nx), int(nx_phys), int(nu)
        route_ix = [int(i) for i in route_ix]
        self.row_ix = tuple(i for i in route_ix if i < self.nx_phys)
        self.field_cols = tuple(route_ix.index(i) for i in self.row_ix)
        p = int(phy_block.free_params.numel())
        E = torch.zeros(self.nx, self.nx_phys, dtype=dtype)
        for i in self.row_ix:
            E[i, i] = 1.0
        self.register_buffer('E', E)
        self.register_buffer('vbar', torch.zeros(p, dtype=dtype))
        self.register_buffer('theta_frozen', torch.zeros(
            p + (1 if self.space == 'affine' else 0), dtype=dtype))

    def expansion_point(self) -> Tensor:
        """The CURRENT free coordinate (attached); the lifecycle detaches it."""
        return self._phy[0].free_params

    @torch.no_grad()
    def set_frozen(self, theta: Tensor, vbar: Tensor) -> None:
        self.theta_frozen.copy_(theta.detach().to(self.theta_frozen.dtype))
        self.vbar.copy_(vbar.detach().to(self.vbar.dtype))

    # CHANGED (D-194): the ONCE-PER-OBJECTIVE half.
    def tangent_pair(self, theta: Tensor):
        """Per-objective physical pass at ``vbar``."""
        direction = theta if self.space == 'tangent' else theta[:-1]
        mats, dmats = self._phy[0].mats_and_tangent_from_free(self.vbar, direction)
        alpha = theta[-1]
        return mats, dmats, alpha

    def forward(self, x: Tensor, u: Tensor, obc_pass=None) -> Tensor:
        """`obc_pass` is the `(mats, dmats)` pair for THIS objective, threaded from `loss`."""
        xb, ub = x[:, : self.nx_phys, :], u
        if obc_pass is None:
            obc_pass = self.tangent_pair(self.theta_frozen)
        mats, dmats = obc_pass[:2]
        fbar, jv = self._phy[0]._rk4_with_tangent(
            xb, torch.zeros_like(xb), ub, mats, dmats)
        corr = jv if self.space == 'tangent' else jv + obc_pass[2] * fbar
        return torch.matmul(self.E, corr)


def make_float64_block(phy_block: Reduced_Gantry_State_Block) -> Reduced_Gantry_State_Block:
    """A float64 copy for the basis build. The live block stays at the pipeline dtype."""
    blk = copy.deepcopy(phy_block)
    blk._cur = None
    return blk.to(torch.float64)


def make_basis_builder(blk64: Reduced_Gantry_State_Block, ref: GantryReferenceSet,
                       row_ix: Sequence[int], chunk: int, device=None,
                       space: str = 'affine'):
    """`vbar64 -> (J, seconds)` for `OBCLifecycle`: float64 jacfwd over the batched step."""
    def build(vbar64: Tensor):
        dev = device if device is not None else vbar64.device
        if next(blk64.buffers()).device != dev:
            blk64.to(dev)
        t0 = time.perf_counter()
        J = build_stacked_sensitivity(lambda v, z: blk64.transition_from_free(v, z),
                                      vbar64.to(dev), ref.Z_step, row_ix, chunk=chunk, device=dev)
        if space == 'affine':
            c = evaluate_step_rows(lambda v, z: blk64.transition_from_free(v, z),
                                   vbar64.to(dev), ref.Z_step, row_ix,
                                   chunk=chunk, device=dev)
            J = torch.cat((J, c[:, None]), dim=1)
        return J, time.perf_counter() - t0
    build.blk64 = blk64
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
        w = ann(cache['Z'])
        return w[:, cache['cols'], 0].reshape(-1)
    return field


def attach_obc(fit_sys, cfg: RunConfig, data, norm, verbose: bool = True):
    """Attach reference set, correction module and lifecycle to a built `fit_sys`."""
    blocks = list(fit_sys.hfn.connected_blocks)
    phy = next((b for b in blocks if isinstance(b, Reduced_Gantry_State_Block)), None)
    nx = fit_sys.hfn.nx
    files = data.train_files
    ref = build_reference_set(cfg, norm, nx_ann=nx - cfg.nx_phys, files=files,
                              verbose=verbose)
    corr = GantryOBCCorrection(phy, nx=nx, nx_phys=cfg.nx_phys, nu=cfg.nu,
                               route_ix=cfg.ann_route_ix, dtype=cfg.dtype_pt,
                               space=cfg.obc_space)
    corr = corr.to(next(phy.parameters()).device)
    blk64 = make_float64_block(phy)
    life = OBCLifecycle(
        reference_field=make_reference_field(ref, corr.field_cols),
        basis_builder=make_basis_builder(blk64, ref, corr.row_ix, chunk=cfg.obc_ref_chunk,
                                         space=cfg.obc_space),
        refresh=cfg.obc_refresh, rtol=cfg.obc_rank_rtol, out_dtype=cfg.dtype_pt,
        expected_rank=int(phy.free_params.numel()) + (cfg.obc_space == 'affine'),
        scale_columns=(cfg.obc_space == 'affine'))
    fit_sys.hfn.obc_correction = corr
    fit_sys.obc = life
    fit_sys.obc_reference = ref
    if verbose:
        print('[obc] attached: protected rows %s (ANN cols %s), %d reference tuples, '
              'space=%s, refresh=%s, solve float64, rollout %s'
              % (corr.row_ix, corr.field_cols, ref.n, cfg.obc_space, cfg.obc_refresh,
                 cfg.dtype_pt))
    return life, corr, ref
