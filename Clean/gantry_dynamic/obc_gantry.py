"""Gantry part of the orthogonal projection: the reference tuples, the basis of the baseline transition, the
network on the tuples and the correction f_obc^n."""
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
from model_augmentation.fit_systems.blocks import IdentifiableGantryStateBlock, Static_ANN_Block
from model_augmentation.fit_systems.obc import OBCProjection, build_stacked_sensitivity, evaluate_step_rows

from .config import RunConfig
from .data import traj_dir, _record_at_fs_new

# the float64 P^-T: measured stage positions y to logical positions q = P^-T y
P_INV_T64 = np.linalg.inv(P.detach().cpu().numpy().astype(np.float64).T)


@dataclass
class GantryReferenceSet:
    """The fixed reference tuples (x_tilde^n_i, x_bar_i = 0, u^n_i) over all training records."""
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
    """qdot_k = (-q_{k+2} + 8 q_{k+1} - 8 q_{k-1} + q_{k-2}) / (12 ts) for k = 2 .. N-3, aligned with q[2:-2].

    # THEORY: fourth-order central difference (Fornberg 1988); the caller uses it only where the full stencil
    # lies inside one record.
    """
    return (-q[4:] + 8.0 * q[3:-1] - 8.0 * q[1:-3] + q[:-4]) / (12.0 * ts)


def build_reference_set(cfg: RunConfig, norm, nx_ann: int, files: Sequence[str],
                        verbose: bool = True) -> GantryReferenceSet:
    """The reference tuples of the training records files."""
    ts = cfg.ts_new
    x_mean = np.asarray(norm.x_mean, dtype=np.float64).reshape(1, 6)
    std_x = np.asarray(norm.std_x, dtype=np.float64).reshape(1, 6)
    u_mean = np.asarray(norm.u_mean, dtype=np.float64).reshape(1, 3)
    std_u = np.asarray(norm.std_u, dtype=np.float64).reshape(1, 3)

    xs, us, rec, samp, xnr = [], [], [], [], []
    for r, f in enumerate(files):
        # one training record in float64, resampled as for training
        d = loadmat(os.path.join(traj_dir(cfg), f), squeeze_me=True)
        u, y = (np.asarray(a, dtype=np.float64) for a in _record_at_fs_new(d, cfg))
        # x_tilde_i = (q_i, qdot_i), q_i = P^-T y_i, at every sample where qdot exists at i and i+1
        q = y @ P_INV_T64.T
        qd = fourth_order_central_difference(q, ts)
        k = np.arange(2, len(u) - 3)
        x_ref = np.hstack([q[k], qd[k - 2]])
        xnr.append(np.hstack([q[k + 1], qd[k - 1]]))
        xs.append(x_ref); us.append(u[k])
        rec.append(np.full(len(k), r)); samp.append(k)
    x_phys = np.concatenate(xs); u_stage = np.concatenate(us)
    record_ix = np.concatenate(rec); sample_ix = np.concatenate(samp)

    x_n = (x_phys - x_mean) / std_x
    u_n = (u_stage - u_mean) / std_u
    N = x_n.shape[0]
    dt = cfg.dtype_pt
    Z_step = torch.from_numpy(np.hstack([x_n, u_n])).to(dt).unsqueeze(-1)
    Z_field = torch.from_numpy(
        np.hstack([x_n, np.zeros((N, nx_ann)), u_n])).to(dt).unsqueeze(-1)
    reference = GantryReferenceSet(
        Z_step=Z_step, Z_field=Z_field,
        record_ix=torch.from_numpy(record_ix).long(), sample_ix=torch.from_numpy(sample_ix).long(),
        record_names=[os.path.basename(f) for f in files],
        x_phys_next_rec=torch.from_numpy(np.concatenate(xnr)))
    if verbose:
        print('[obc] reference set: %d tuples from %d records' % (N, len(files)))
    return reference


class GantryOBCCorrection(nn.Module):
    """f_obc^n = E (varphi(beta_lin) a + c f_base^n(beta_lin)) at the current state and input, theta_aux = (a, c)."""

    def __init__(self, phy_block: IdentifiableGantryStateBlock, nx: int, nx_phys: int,
                 route_ix: Sequence[int], dtype: torch.dtype):
        super().__init__()
        self.space = 'affine'
        self._phy = [phy_block]
        self.nx, self.nx_phys = int(nx), int(nx_phys)
        route_ix = [int(i) for i in route_ix]
        self.row_ix = tuple(i for i in route_ix if i < self.nx_phys)
        self.field_cols = tuple(route_ix.index(i) for i in self.row_ix)
        p = int(phy_block.free_params.numel())
        # E: the protected physical rows of the model state
        E = torch.zeros(self.nx, self.nx_phys, dtype=dtype)
        for i in self.row_ix:
            E[i, i] = 1.0
        self.register_buffer('E', E)
        self.register_buffer('vbar', torch.zeros(p, dtype=dtype))
        # theta_aux = (a, c): p sensitivity coefficients and the coefficient of the baseline transition
        self.register_buffer('theta_frozen', torch.zeros(p + 1, dtype=dtype))

    def expansion_point(self) -> Tensor:
        """The current training coordinate beta, attached; OBCProjection.refresh detaches it."""
        return self._phy[0].free_params

    @torch.no_grad()
    def set_frozen(self, theta: Tensor, vbar: Tensor) -> None:
        """Store theta_aux and beta_lin for the prediction path."""
        self.theta_frozen.copy_(theta.detach().to(self.theta_frozen.dtype))
        self.vbar.copy_(vbar.detach().to(self.vbar.dtype))

    def tangent_pair(self, theta: Tensor):
        """(matrices at beta_lin, their tangent in the direction a, c) for theta_aux = (a, c), once per objective."""
        mats, dmats = self._phy[0].mats_and_tangent_from_free(self.vbar, theta[:-1])
        alpha = theta[-1]
        return mats, dmats, alpha

    def forward(self, x: Tensor, u: Tensor, obc_pass=None) -> Tensor:
        """f_obc^n at (x, u); obc_pass is the tangent_pair of this objective, or None for theta_frozen."""
        xb, ub = x[:, : self.nx_phys, :], u
        if obc_pass is None:
            obc_pass = self.tangent_pair(self.theta_frozen)
        mats, dmats = obc_pass[:2]
        # f_base^n(beta_lin) and its directional derivative varphi(beta_lin) a, from a zero state tangent
        fbar, jv = self._phy[0]._rk4_with_tangent(
            xb, torch.zeros_like(xb), ub, mats, dmats)
        return torch.matmul(self.E, jv + obc_pass[2] * fbar)


def make_float64_block(phy_block: IdentifiableGantryStateBlock) -> IdentifiableGantryStateBlock:
    """A float64 copy of the physics block for the basis build; the live block keeps the training dtype."""
    return copy.deepcopy(phy_block).to(torch.float64)


def make_basis_builder(blk64: IdentifiableGantryStateBlock, ref: GantryReferenceSet,
                       row_ix: Sequence[int], chunk: int, device=None,
                       space: str = 'affine'):
    """vbar64 -> (J, seconds) for OBCProjection, J = [Phi F_base^n(beta_lin)] in float64 on the reference tuples."""
    def build(vbar64: Tensor):
        dev = device if device is not None else vbar64.device
        if next(blk64.buffers()).device != dev:
            blk64.to(dev)
        t0 = time.perf_counter()
        J = build_stacked_sensitivity(blk64.transition_from_free,
                                      vbar64.to(dev), ref.Z_step, row_ix, chunk=chunk, device=dev)
        if space == 'affine':
            # the extra column is the baseline transition F_base^n(beta_lin) itself, not the offset
            # Gamma = F_base^n(beta_lin) - Phi beta_lin: the same span, so the same projection, but other
            # coordinates: with coefficients (a, c) here, those on [Phi Gamma] are (a + beta_lin c, c)
            c = evaluate_step_rows(blk64.transition_from_free,
                                   vbar64.to(dev), ref.Z_step, row_ix,
                                   chunk=chunk, device=dev)
            J = torch.cat((J, c[:, None]), dim=1)
        return J, time.perf_counter() - t0
    return build


def make_reference_field(ref: GantryReferenceSet, field_cols: Sequence[int]):
    """ann -> F: the network on the reference tuples, its routed physical columns stacked sample-major, with
    autograd."""
    cols = torch.as_tensor(list(field_cols), dtype=torch.long)
    # the reference inputs, cached per device and dtype
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


def attach_obc(fit_sys, cfg: RunConfig, data, norm):
    """Attach the reference set, the correction module and the projection to a built model."""
    phy = fit_sys.hfn.connected_blocks[0]
    nx = fit_sys.hfn.nx
    reference = build_reference_set(cfg, norm, nx_ann=nx - cfg.nx_phys, files=data.train_files)
    correction = GantryOBCCorrection(phy, nx=nx, nx_phys=cfg.nx_phys, route_ix=cfg.ann_route_ix,
                                     dtype=cfg.dtype_pt)
    correction = correction.to(next(phy.parameters()).device)
    baseline64 = make_float64_block(phy)
    projection = OBCProjection(
        reference_field=make_reference_field(reference, correction.field_cols),
        basis_builder=make_basis_builder(baseline64, reference, correction.row_ix, chunk=cfg.obc_ref_chunk,
                                         space=cfg.obc_space))
    fit_sys.hfn.obc_correction = correction
    fit_sys.obc = projection
    fit_sys.obc_reference = reference
    print('[obc] attached: protected rows %s (ANN cols %s), %d reference tuples, '
          'space=%s, refresh=%s, solve float64, rollout %s'
          % (correction.row_ix, correction.field_cols, reference.n, cfg.obc_space, cfg.obc_refresh,
             cfg.dtype_pt))
