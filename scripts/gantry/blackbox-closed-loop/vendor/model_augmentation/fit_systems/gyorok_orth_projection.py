"""Orthogonal-projection regularisation exactly as `gyorok2025l4dc` releases it (D-185).

This module reimplements the method of Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth (L4DC 2025)
as the released code in `orthogonal-augmentation-main/` defines it, for the gantry closed loop at
`nx_ann = 0`. It is deliberately NOT the routed, rank-truncated, data-state variant in
`orth_projection.py`, and NOT the windowed trajectory method in `trajectory_orth_projection.py`.
Those stay untouched; this file exists so that "Gyorok's method, our system" can be read off
without translation.

WHAT THE REFERENCE DOES (all read from the released code, 2026-09-09):

  regressor   `calculate_orth_matrix(x, u) -> F(x,u)` of shape `(Nd*Nx, Np)`, per system
              (`system_models.py:141`). Sample-major: rows `[i*Nx, (i+1)*Nx)` belong to sample i.
  basis       `U1, _, _ = torch.linalg.svd(Matrix, full_matrices=False)` with NO truncation
              (`utils.py:83`). Every column is kept.
  penalty     `orthLambda * || U1^T x_net ||^2`, a SUM over the fixed point set, no `/N`
              (`augmentation_structures.py:152-153`).
  point set   fixed and precomputed, passed once through `loss_kwargs` (`fit_system.py:47`);
              built either from measured states (`x_meas=True`) or, per the paragraph before Remark 2 (Sect. 3, p. 7), from an FP-model
              simulation (`utils.py:62`). The gantry is that FP-simulated case, closed loop.

WHAT IS GANTRY-SPECIFIC and lives here rather than in the physical block: the baseline is
nonlinear in `theta`, so the paper's Sect. 4 extended regressor applies,
`F = [Phi | Gamma]`, `Phi = d f_theta / d theta` at `theta_bar`, `Gamma = f_theta_bar - Phi theta_bar`.
`Parameterized_Gantry_State_Block` trains `log theta`, so the Jacobian is taken in the log
coordinate and divided by `theta_bar` (chain rule, exact). Column scaling does not change
`span(F)`, hence not `U1`; it changes the singular values, which is why `spectrum_report`
reports both scalings and the truncation decision names the one it used.

Contents:
  gantry_orth_matrix     the `calculate_orth_matrix` for the gantry block (Phi, Gamma, f)
  finite_difference_check  central-difference validation of Phi on a few samples
  orth_basis             Gyorok's SVD basis, optionally truncated (truncation is REPORTED)
  spectrum_report        rank at several tolerances, gaps, both column scalings
  GyorokOrthPenalty      beta * ||U1^T f_ANN(X,U)||^2 on the fixed point set (sum convention)
"""
__project_origin__ = "added"

import numpy as np
import torch
from torch import nn, Tensor
from torch.func import functional_call, jacfwd, vmap

_F64 = torch.float64


# ------------------------------------------------------------------------- regressor
def _block_step(blk, log_params: Tensor, z: Tensor) -> Tensor:
    """One physical-block step with `log_params` supplied functionally. z (B, nx+nu, 1)."""
    return functional_call(blk, {'log_params': log_params}, (z,))


def gantry_orth_matrix(blk, X: Tensor, U: Tensor, chunk: int = 512, verbose: bool = False):
    """Gyorok's `calculate_orth_matrix` for the gantry physical block.

    Parameters
    ----------
    blk : Parameterized_Gantry_State_Block, float64, eval mode, `log_params` at the anchor
        (zeros give `theta_bar = params_init`).
    X : (Nd, nx) NORMALISED physical states, U : (Nd, nu) NORMALISED inputs, as the block sees
        them inside the interconnect. Under closed loop `U` must be `u_data + u_fb`, the input
        the block was actually stepped with, not the recorded plant input alone.

    Returns
    -------
    Phi   (Nd*nx, n_theta)  d f / d theta at theta_bar, RAW theta units, sample-major
    Gamma (Nd*nx, 1)        f(theta_bar) - Phi theta_bar   (paper Sect. 4 constant term)
    F     (Nd*nx, 1)        f(theta_bar) itself, kept for diagnostics
    theta_bar (n_theta,)
    """
    X = torch.as_tensor(X, dtype=_F64)
    U = torch.as_tensor(U, dtype=_F64)
    Nd, nx = X.shape
    nu = U.shape[1]
    lp = blk.log_params.detach().clone()
    theta_bar = blk._recover_params().detach()
    n_theta = theta_bar.numel()

    def f_single(lp_, z_):                      # z_ (nx+nu, 1) -> (nx,)
        return _block_step(blk, lp_, z_.unsqueeze(0)).reshape(nx)

    # Forward-mode over the n_theta parameter directions, vmapped over the samples: n_theta
    # jvps per chunk instead of Nd*nx vjps. The block rebuilds its M(Y) structure per call and
    # caches it in `_cur` as a functorch wrapper; that cache is per-forward and is overwritten
    # by the next plain call, so nothing here leaks into the block's state.
    jac_fn = vmap(jacfwd(f_single, argnums=0), in_dims=(None, 0))
    Phi = torch.empty(Nd, nx, n_theta, dtype=_F64)
    Fv = torch.empty(Nd, nx, dtype=_F64)
    with torch.no_grad():
        for s in range(0, Nd, chunk):
            z = torch.cat([X[s:s + chunk], U[s:s + chunk]], dim=1).unsqueeze(-1)
            Phi[s:s + chunk] = jac_fn(lp, z)
            Fv[s:s + chunk] = _block_step(blk, lp, z).reshape(-1, nx)
            if verbose and (s // chunk) % 10 == 0:
                print(f'[gyorok] jacobian {min(s + chunk, Nd)}/{Nd}', flush=True)
    blk._cur = None
    # d/d theta = (d/d log theta) / theta   (exact chain rule at the anchor)
    Phi = (Phi / theta_bar.view(1, 1, n_theta)).reshape(Nd * nx, n_theta)
    Fv = Fv.reshape(Nd * nx, 1)
    Gamma = Fv - Phi @ theta_bar.view(n_theta, 1)
    return Phi, Gamma, Fv, theta_bar


def finite_difference_check(blk, X: Tensor, U: Tensor, Phi: Tensor, n_samples: int = 5,
                            rel_step: float = 1e-6):
    """Central-difference check of Phi on the first `n_samples` samples.

    Perturbs each `theta_i` by `rel_step * theta_i` (in log space that is `log(1 +- rel_step)`)
    and compares `(f(theta+) - f(theta-)) / (2 delta)` with the autograd column. Returns the
    worst relative column error over the checked samples and the per-column table.
    """
    X = torch.as_tensor(X[:n_samples], dtype=_F64)
    U = torch.as_tensor(U[:n_samples], dtype=_F64)
    nx = X.shape[1]
    lp0 = blk.log_params.detach().clone()
    theta = blk._recover_params().detach()
    n_theta = theta.numel()
    z = torch.cat([X, U], dim=1).unsqueeze(-1)
    Phi_s = Phi[:n_samples * nx].reshape(n_samples, nx, n_theta)
    fd = torch.empty_like(Phi_s)
    with torch.no_grad():
        for i in range(n_theta):
            lp_p, lp_m = lp0.clone(), lp0.clone()
            lp_p[i] += np.log1p(rel_step)
            lp_m[i] += np.log1p(-rel_step)
            fp = _block_step(blk, lp_p, z).reshape(n_samples, nx)
            fm = _block_step(blk, lp_m, z).reshape(n_samples, nx)
            delta = theta[i] * (np.expm1(np.log1p(rel_step)) - np.expm1(np.log1p(-rel_step))) / 2
            fd[:, :, i] = (fp - fm) / (2 * delta)
    blk._cur = None
    num = torch.linalg.vector_norm(fd - Phi_s, dim=(0, 1))
    den = torch.linalg.vector_norm(Phi_s, dim=(0, 1))
    rel = (num / den.clamp(min=1e-300)).numpy()
    return float(np.max(rel)), rel, den.numpy()


# --------------------------------------------------------------------------- basis
def orth_basis(F: Tensor, rank_rtol: float = 0.0):
    """Gyorok's `U1`: left singular vectors of the regressor.

    `rank_rtol = 0` keeps every column, which is the released code. A positive value truncates
    at `sigma_k > rank_rtol * sigma_0`; the choice is the caller's and must be reported with the
    spectrum that justified it (D-185).
    """
    F = torch.as_tensor(F, dtype=_F64)
    U1, S, _ = torch.linalg.svd(F, full_matrices=False)
    keep = int(torch.sum(S > rank_rtol * S[0])) if rank_rtol > 0 else S.numel()
    return U1[:, :keep], S, keep


def spectrum_report(Phi: Tensor, Gamma: Tensor, theta_bar: Tensor,
                    tols=(1e-6, 1e-8, 1e-10, 1e-12, 1e-14)):
    """Singular spectra and rank verdicts for the regressor under both column scalings.

    `raw`      F = [Phi | Gamma], Gyorok's units (d f / d theta, theta in SI).
    `log`      F = [Phi diag(theta_bar) | Gamma], i.e. d f / d log theta: unit relative
               parameter perturbations. Same span, different singular values.
    Both are reported with and without the Gamma column, since Gyorok's Eq. 8 form has no
    constant term and Sect. 4's has one.
    """
    Phi = torch.as_tensor(Phi, dtype=_F64)
    Gamma = torch.as_tensor(Gamma, dtype=_F64)
    th = theta_bar.view(1, -1)
    variants = {
        'raw_phi': Phi,
        'raw_ext': torch.cat([Phi, Gamma], dim=1),
        'log_phi': Phi * th,
        'log_ext': torch.cat([Phi * th, Gamma], dim=1),
    }
    out = {}
    for name, M in variants.items():
        S = torch.linalg.svdvals(M).numpy()
        ratios = S / S[0]
        gaps = np.log10(ratios[:-1] / np.clip(ratios[1:], 1e-300, None))
        out[name] = dict(
            n_rows=int(M.shape[0]), n_cols=int(M.shape[1]),
            singular_values=[float(v) for v in S],
            ratio_to_sigma0=[float(v) for v in ratios],
            rank_at_rtol={f'{t:.0e}': int(np.sum(ratios > t)) for t in tols},
            largest_gap_decades=float(np.max(gaps)),
            largest_gap_after_index=int(np.argmax(gaps)) + 1,
            log10_gaps=[float(g) for g in gaps],
        )
    return out


# --------------------------------------------------------------------------- penalty
class GyorokOrthPenalty(nn.Module):
    """beta * || U1^T stack(f_ANN(Z_pts)) ||^2, Gyorok's sum convention, on a fixed point set.

    Parameters
    ----------
    U1 : (Nd*nx, k)   basis from `orth_basis` (k = n_cols when untruncated)
    Z_pts : (Nd, nx_phys+nx_ann+nu, 1)  fixed NORMALISED ANN inputs at the point set,
        `[x_phys; x_aug = 0; u]`. At `nx_ann = 0` the output rows are the six physical rows
        and pre- and post-routing coincide (D-185). With latent states the penalty is
        evaluated on the `x_aug = 0` slice, which is where the FP rollout lives; what the
        network does off that slice is tracked separately, not penalised.
    beta : float      swept, never computed (paper Remark 1, Fig. 3)
    """

    def __init__(self, U1, Z_pts, beta: float, dtype=torch.float32, route_cols=None):
        super().__init__()
        self.register_buffer('U1', torch.as_tensor(U1, dtype=dtype))
        self.register_buffer('Z_pts', torch.as_tensor(Z_pts, dtype=dtype))
        self.beta = float(beta)
        # Same read-only surface as `orth_projection.OrthProjectionPenalty` (`Q`, `route_cols`),
        # so the pipeline's joint-estimation probe (gantry_dynamic/training.py) can log the
        # negating fraction ||Q^T f||^2 / ||f||^2 for this penalty without a second code path.
        # `route_cols` are the ANN OUTPUT columns that write physical rows, in sample-major
        # order matching the regressor rows. At nx_ann = 0 that is every column; with latent
        # states the latent columns carry zero baseline sensitivity and are NOT penalised: the
        # paper's method reaches the direct route only (D-185, latent-route tracker).
        if self.U1.shape[0] % self.Z_pts.shape[0]:
            raise ValueError('U1 rows must be a multiple of the point count')
        n_rows = self.U1.shape[0] // self.Z_pts.shape[0]
        self.route_cols = list(range(n_rows)) if route_cols is None else list(route_cols)
        if len(self.route_cols) != n_rows:
            raise ValueError('route_cols (%d) must match the regressor rows per sample (%d)'
                             % (len(self.route_cols), n_rows))

    @property
    def Q(self) -> Tensor:
        return self.U1

    def penalty_of_field(self, f_stacked: Tensor) -> Tensor:
        return self.beta * torch.linalg.vector_norm(self.U1.T @ f_stacked) ** 2

    def forward(self, ann_block) -> Tensor:
        w = ann_block(self.Z_pts)                 # (Nd, nw, 1)
        return self.penalty_of_field(w[:, self.route_cols, 0].reshape(-1))
