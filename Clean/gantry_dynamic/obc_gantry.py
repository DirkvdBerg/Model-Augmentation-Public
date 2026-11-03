"""Orthogonal projection (OBC) for the gantry: removes from the network's correction what the baseline parameters
beta could also give, so beta keeps its physical meaning. Superscript n = normalised.

  tuples     (x_tilde_i, xbar_i = 0, u_i): measured state, zero added states and measured force at training
             sample i; x_tilde_i = (q_i, qdot_i), q_i = P^-T y_i
  beta_lin   beta at the start of the epoch
  varphi     d f_base^n(theta_base(beta), x_tilde^n, u^n) / d beta at beta_lin (6 x 10); Phi = col(varphi)
  J          [Phi  col(f_base^n(beta_lin))], stacked over the tuples
  F^n_aug    col(f^n_aug(theta_aug, (x_tilde^n_i, 0), u^n_i)), the network's correction on the tuples
  theta_aux  (a, c) = (J^T J)^-1 J^T F^n_aug, the least-squares coefficients of F^n_aug on J (solved in obc.py)
  f_obc^n    varphi(beta_lin) a + c f_base^n(beta_lin) on the six physical rows: the part of the network's
             correction that beta could also give; subtracted at every step

The thesis uses [Phi Gamma], Gamma = col(f_base^n(beta_lin) - varphi beta_lin). J appends f_base^n(beta_lin)
itself, which the Jacobian build already evaluates; both span the same space, so f_obc^n is the same and only
theta_aux differs, (a + c beta_lin, c) in the thesis.
"""
__project_origin__ = "added"

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from model_augmentation.fit_systems.blocks import IdentifiableGantryStateBlock, Static_ANN_Block
from model_augmentation.fit_systems.obc import OBCProjection, build_stacked_sensitivity
from model_augmentation.systems.gantry_ss import P_INV_T

from .config import RunConfig


@dataclass
class GantryReferenceSet:
    """The fixed OBC tuples of the training records, normalised: baseline_input = (x_tilde^n_i, u^n_i), from which
    J is built; network_input = (x_tilde^n_i, 0, u^n_i), from which F^n_aug is computed."""
    baseline_input: Tensor
    network_input: Tensor


def fourth_order_central_difference(q: np.ndarray, ts: float) -> np.ndarray:
    """Velocity of the OBC tuples; at the 212 Hz absorber mode its error is 0.04 %, against 1.8 % for a
    second-order central difference.

    qdot_i = (-q_{i+2} + 8 q_{i+1} - 8 q_{i-1} + q_{i-2}) / (12 T_s); qdot_i needs the two samples before and after
    i, so row j of the result is the velocity at sample i = j + 2.

    Args:
        q: (N, 3) generalised positions [m, rad, m].
        ts: sample time T_s [s].
    Returns:
        (N - 4, 3) qdot [m/s, rad/s, m/s].
    """
    return (-q[4:] + 8.0 * q[3:-1] - 8.0 * q[1:-3] + q[:-4]) / (12.0 * ts)


def build_reference_set(cfg: RunConfig, norm, records: Sequence) -> GantryReferenceSet:
    """OBC tuples of the training records, at i = 2 .. N - 4: the samples where qdot can be computed at i and at
    i + 1.
    Built once before training from the measured data, not from the simulated states, so J and F^n_aug are evaluated
    at the same points throughout training.

    Args:
        cfg: run configuration.
        norm: means and standard deviations of the training records (Norm).
        records: the training records at fs_new (Datasets.train_list).
    Returns:
        GantryReferenceSet: baseline_input (N, 9, 1) and network_input (N, 9 + n_a, 1), normalised.
    """
    x_mean = np.asarray(norm.x_mean, dtype=cfg.dtype_np).reshape(1, cfg.nx_phys)
    std_x = np.asarray(norm.std_x, dtype=cfg.dtype_np).reshape(1, cfg.nx_phys)
    u_mean = np.asarray(norm.u_mean, dtype=cfg.dtype_np).reshape(1, cfg.nu)
    std_u = np.asarray(norm.std_u, dtype=cfg.dtype_np).reshape(1, cfg.nu)

    # 1. Tuples per record, in physical units.
    xs, us = [], []
    for record in records:
        u, y = np.asarray(record.u, dtype=cfg.dtype_np), np.asarray(record.y, dtype=cfg.dtype_np)
        # x_tilde_i = (q_i, qdot_i), q_i = P^-T y_i, i = 2 .. N - 4; one sample per row, so q = y @ P_INV_T.T
        q = y @ P_INV_T.T
        # qdot by central differences on the positions filtered by the anti-alias filter of data.py (zero phase,
        # 1.6 kHz), so the noise above 1.6 kHz is not amplified by the difference
        qdot = fourth_order_central_difference(q, cfg.ts_new)
        i = np.arange(2, len(u) - 3)
        # row i - 2 of qdot is the velocity at sample i
        xs.append(np.hstack([q[i], qdot[i - 2]]))
        us.append(u[i])
    x_tilde = np.concatenate(xs)
    u_act = np.concatenate(us)

    # 2. Normalise: x_tilde^n = (x_tilde - mu_x) / sigma_x, u^n = (u_act - mu_u) / sigma_u.
    x_n = (x_tilde - x_mean) / std_x
    u_n = (u_act - u_mean) / std_u
    N = x_n.shape[0]
    # stack per tuple, as column vectors: baseline input (x_tilde^n_i, u^n_i), (N, 9, 1); network input
    # (x_tilde^n_i, 0, u^n_i), (N, 9 + n_a, 1)
    baseline_input = torch.from_numpy(np.hstack([x_n, u_n])).to(cfg.dtype_pt).unsqueeze(-1)
    network_input = torch.from_numpy(
        np.hstack([x_n, np.zeros((N, cfg.n_a)), u_n])).to(cfg.dtype_pt).unsqueeze(-1)
    print('[obc] reference set: %d tuples from %d records' % (N, len(records)))
    return GantryReferenceSet(baseline_input=baseline_input, network_input=network_input)


class GantryOBCCorrection(nn.Module):
    """The OBC correction f_obc^n at the current state and input, subtracted from the network's correction of the
    six physical states at every step. Training computes theta_aux at
    every training step on the basis J, which is rebuilt at beta_lin once per epoch; validation, polish, PS2 and
    evaluation use the copy of theta_aux and beta_lin stored here."""

    def __init__(self, baseline_block: IdentifiableGantryStateBlock, nx: int, nx_phys: int, dtype: torch.dtype,
                 reference: GantryReferenceSet, chunk: int):
        """Create the stored beta_lin and theta_aux, which are zero until the first set_theta_aux, and hold the
        OBC tuples.

        Args:
            baseline_block: the identifiable baseline block, which holds beta and computes f_base^n.
            nx: model state size 6 + n_a.
            nx_phys: number of physical states (6).
            dtype: dtype of beta_lin and theta_aux (cfg.dtype_pt).
            reference: the OBC tuples.
            chunk: tuples per Jacobian evaluation; sets the memory use, not the result.
        """
        super().__init__()
        # the baseline matrices (K, C, parts of M(Y)^-1) at beta_lin, and their change when beta moves in the
        # direction a, theta_aux = (a, c)
        self.baseline_matrices_and_derivative = baseline_block.mats_and_tangent_from_free
        # one baseline step with those matrices: f_base^n(beta_lin) and varphi(beta_lin) a
        self.baseline_step_and_derivative = baseline_block.rk4_with_tangent
        self.baseline_step = baseline_block.transition_from_free
        self.chunk = chunk
        self.nx, self.nx_phys = nx, nx_phys
        # network output i is the correction of physical state i
        self.physical_rows = tuple(range(self.nx_phys))
        n_beta = baseline_block.free_params.numel()
        # beta_lin: the beta that J was built at
        self.register_buffer('beta_lin', torch.zeros(n_beta, dtype=dtype))
        # theta_aux = (a, c), the coefficients of f_obc^n = varphi(beta_lin) a + c f_base^n(beta_lin)
        self.register_buffer('theta_aux', torch.zeros(n_beta + 1, dtype=dtype))
        # the OBC tuples, on the model's device (CPU or GPU) because J and F^n_aug are evaluated on them; not saved
        # in the checkpoint, since build_reference_set rebuilds them from the data
        self.register_buffer('baseline_input', reference.baseline_input, persistent=False)
        self.register_buffer('network_input', reference.network_input, persistent=False)

    @torch.no_grad()
    def set_theta_aux(self, theta_aux: Tensor, beta_lin: Tensor) -> None:
        """Store theta_aux and the beta_lin of its basis, for validation, polish, PS2 and evaluation."""
        self.theta_aux.copy_(theta_aux)
        self.beta_lin.copy_(beta_lin)

    def correction_matrices(self, theta_aux: Tensor):
        """Matrices for forward, built once per rollout because they do not depend on the state.

        varphi(beta_lin) a is the derivative of f_base^n in the direction a. f_base^n depends on beta only through
        its matrices, so forward needs the matrices at beta_lin and their derivative in the direction a.

        Args:
            theta_aux: (11,) (a, c).
        Returns:
            mats: the baseline matrices at beta_lin.
            dmats: the derivative of mats in the direction a.
            c: the coefficient of f_base^n.
        """
        mats, dmats = self.baseline_matrices_and_derivative(self.beta_lin, theta_aux[:-1])
        return mats, dmats, theta_aux[-1]

    def forward(self, x: Tensor, u: Tensor, obc_pass) -> Tensor:
        """OBC correction f_obc^n at the current state and input; subtracting it makes the network's correction
        orthogonal to J on the tuples.

        Args:
            x: (B, nx, 1) x_hat^n_k, the normalised model state (physical and added) at step k; B the number of
                rollouts simulated in parallel.
            u: (B, 3, 1) u_hat^n_k, the normalised closed-loop input at step k.
            obc_pass: (mats, dmats, c) from correction_matrices.
        Returns:
            (B, nx, 1) f_obc^n on the six physical rows, zero on the n_a added rows.
        """
        # x_tilde^n_k: the six physical rows of x_hat^n_k
        x_tilde = x[:, : self.nx_phys, :]
        mats, dmats, c = obc_pass
        # f_base^n(beta_lin) and varphi(beta_lin) a, the derivative of f_base^n in the direction a, in one RK4 step
        # from mats and dmats; x_tilde^n_k is an input of the step, so its derivative is zero (torch.zeros_like)
        f_base, varphi_a = self.baseline_step_and_derivative(
            x_tilde, torch.zeros_like(x_tilde), u, mats, dmats)
        # f_obc^n = varphi a + c f_base^n on the six physical rows, zero on the n_a added rows
        n_a = self.nx - self.nx_phys
        return torch.nn.functional.pad(varphi_a + c * f_base, (0, 0, 0, n_a))

    def basis(self, beta_lin: Tensor) -> Tensor:
        """J at beta_lin: at every tuple, how the baseline step changes with each of the 10 parameters, and the
        step itself."""
        # J = [Phi  col(f_base^n(beta_lin, x_tilde^n_i, u^n_i))], (6 N, 11)
        return build_stacked_sensitivity(self.baseline_step, beta_lin, self.baseline_input, self.physical_rows,
                                         chunk=self.chunk, append_value=True)

    def network_field(self, ann: Static_ANN_Block) -> Tensor:
        """F^n_aug: the network's correction of the six physical states at every tuple, in the row order of J."""
        # F^n_aug = col(f^n_aug(theta_aug, (x_tilde^n_i, 0), u^n_i)), (6 N,)
        return ann(self.network_input)[:, list(self.physical_rows), 0].reshape(-1)


def attach_obc(fit_sys, baseline_block: IdentifiableGantryStateBlock, cfg: RunConfig, data, norm):
    """Give the model the OBC correction f_obc^n (subtracted at every step) and the projection that computes it.

    Args:
        fit_sys: the augmented model; gets hfn.obc_correction and obc.
        baseline_block: the identifiable baseline block of the model.
        cfg: run configuration.
        data: the records (Datasets); the tuples come from the training records.
        norm: means and standard deviations of the training records (Norm).
    """
    reference = build_reference_set(cfg, norm, records=data.train_list)
    correction = GantryOBCCorrection(baseline_block, nx=fit_sys.hfn.nx, nx_phys=cfg.nx_phys, dtype=cfg.dtype_pt,
                                     reference=reference, chunk=cfg.obc_ref_chunk)
    fit_sys.hfn.obc_correction = correction
    fit_sys.obc = OBCProjection(reference_field=correction.network_field, basis_builder=correction.basis)
