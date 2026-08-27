"""Builder for the orthogonal-projection penalty (plan D7.4, revised; D-111).

Constructs the fixed penalty object for a run configuration: DATA-DERIVED
states of all training records (q = P^-T y exactly, qdot by finite
differencing -- the same construction as data.py compute_normalization),
parameter Jacobian stack at theta_bar (the run's params_init anchor, D7.5) on
the decimated sample set (restricted to the routed physical rows), extended
regressor [Phi | Gamma], economy SVD, rank truncation, and the fixed ANN input
points Z_pts = [x_phys, x_aug=0, u].

States are data-derived, NOT FP-simulated (D-111): the paper's simulated-
states fallback (GYOROK p. 7) assumes the FP rollout stays near the data
manifold, which fails for the marginally stable K=0 axes over long records --
measured: worst negation-signature leakage 0.164 with rollout states (step7b)
vs 0.017 with data/truth states (step7c). Data-derived states realize the
paper's PRIMARY full-state-measurement setting (their Sect. 3 assumption and
their code's x_meas=True) via the static output inversion. Note the states
therefore carry the true system's imprint (hidden-absorber effect ~1e-5 m on
y), which is exactly the real-machine situation. ANN inputs use x_aug = 0
(no pre-training absorber estimate exists; the baseline regressor does not
depend on the absorber states).

Everything is computed in float64 (Stage A convention) and cast to the
pipeline dtype inside OrthProjectionPenalty (D7.7). Results are cached to an
npz keyed by the constructing configuration (including states='data', so
pre-D-111 rollout-based caches can never be loaded); reruns load instead of
recompute (measured fresh-build cost at stride 100: ~6 min Jacobians +
~0.03 s SVD; the ~137 s rollout is gone with D-111).

Validation trail: scripts/gantry/orth-projection/ steps 0-6 validate the
constructions (step1 Jacobian, step2 stacking/SVD, step3 subspace, step4
rank, step6 penalty); step7b/7c justify the state source.
"""
__project_origin__ = "added"

import os
import json
import time
import hashlib

import numpy as np
import torch

from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
from model_augmentation.fit_systems.orth_projection import OrthProjectionPenalty
from model_augmentation.systems.gantry_ss import P

from .config import RunConfig, save_dir

_F64 = torch.float64


def theta_bar_for(cfg: RunConfig) -> np.ndarray:
    """theta_bar = the run's params_init anchor (D7.5): nominal, detuned if set."""
    from model_augmentation.systems import gantry_ss as gss
    names = Parameterized_Gantry_State_Block.PARAM_NAMES
    nominal = np.array([float(getattr(gss, n)) for n in names])
    if cfg.joint_estimation and cfg.param_init_detune is not None:
        return nominal * np.asarray(cfg.param_init_detune, dtype=float)
    return nominal


def _cache_path(cfg: RunConfig, theta_bar) -> str:
    payload = dict(mode=cfg.mode, fs=cfg.fs_new_hz, stride=cfg.orth_point_stride,
                   route=list(cfg.ann_route_ix), up=cfg.up_sample,
                   theta=[round(float(t), 10) for t in theta_bar],
                   rank_tol=cfg.orth_rank_tol, nx_ann=cfg.nx_ann,
                   na_nb=cfg.na_nb,
                   states='data')   # D-111: excludes pre-revision rollout caches
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    cdir = os.path.join(save_dir(cfg), 'orth_cache')
    os.makedirs(cdir, exist_ok=True)
    return os.path.join(cdir, f'orth_{key}.npz')


def _x_logical_from_data(sd) -> np.ndarray:
    """Data-derived [q, qdot]: P^-T on y + forward-FD velocities (as data.py)."""
    fs = 1.0 / sd.dt
    P_inv_T = np.linalg.inv(P.numpy().T).astype(np.float64)
    pos = (P_inv_T @ sd.y.T).T
    vel = np.diff(pos, axis=0) * fs
    vel = np.vstack([vel[:1], vel])
    return np.hstack([pos, vel])


def _make_block(cfg, norm, theta_bar):
    blk = Parameterized_Gantry_State_Block(
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean,
        Ts=cfg.ts_new, up_sample=cfg.up_sample,
        params_init=torch.tensor(theta_bar, dtype=_F64)).to(_F64)
    blk.eval()
    return blk


def _compute(cfg: RunConfig, data, norm, theta_bar, verbose=True):
    """Fresh build: data-derived states -> decimated Jacobian stack at theta_bar -> SVD."""
    NX, nu = cfg.nx_phys, cfg.nu
    K0 = cfg.na_nb
    blk = _make_block(cfg, norm, theta_bar)
    theta_t = torch.tensor(theta_bar, dtype=_F64)

    # States from measured outputs (D-111): q = P^-T y (exact static inversion),
    # qdot by FD -- no model integration, no drift, no theta dependence.
    # Same K0 trim and record order as the u stack.
    u_norms = [((sd.u[K0:] - norm.u_mean.flatten()) / norm.std_u.flatten()).astype(np.float64)
               for sd in data.train_list]
    X_flat = np.concatenate([
        ((_x_logical_from_data(sd)[K0:] - norm.x_mean.flatten())
         / norm.std_x.flatten()).astype(np.float64)
        for sd in data.train_list])
    U_flat = np.concatenate(u_norms)
    if verbose:
        print(f'[orth] data-derived states: {len(data.train_list)} records, '
              f'{X_flat.shape[0]} samples')

    # Decimated Jacobian stack, routed physical rows (the guard: from config).
    r_phys = [i for i in cfg.ann_route_ix if i < NX]
    n_r = len(r_phys)
    sub_ix = np.arange(0, X_flat.shape[0], cfg.orth_point_stride)
    Phi = np.empty((len(sub_ix) * n_r, 14), dtype=np.float64)
    Gam = np.empty((len(sub_ix) * n_r, 1), dtype=np.float64)
    t0 = time.time()
    for si, ix in enumerate(sub_ix):
        x6 = torch.tensor(X_flat[ix], dtype=_F64)
        u3 = torch.tensor(U_flat[ix], dtype=_F64)
        zin = torch.cat([x6.view(1, NX, 1), u3.view(1, 3, 1)], dim=1)
        xn = blk.nonlinear_function(zin).view(NX)
        J = torch.zeros(NX, 14, dtype=_F64)
        for r in range(NX):
            g = torch.autograd.grad(xn[r], blk.log_params, retain_graph=(r < NX - 1))[0]
            J[r] = g
        J_theta = (J / theta_t.view(1, 14)).detach()
        Jr = J_theta[r_phys, :]
        Phi[si * n_r:(si + 1) * n_r, :] = Jr.numpy()
        Gam[si * n_r:(si + 1) * n_r, 0] = (xn.detach()[r_phys] - Jr @ theta_t).numpy()
        if verbose and (si + 1) % 1000 == 0:
            rate = (si + 1) / (time.time() - t0)
            print(f'[orth] jac {si+1}/{len(sub_ix)}  {rate:.1f} samples/s', flush=True)
    if verbose:
        print(f'[orth] jacobians: {time.time()-t0:.1f} s')

    Phi_tilde = np.hstack([Phi, Gam])
    Q_full, S, _ = np.linalg.svd(Phi_tilde, full_matrices=False)
    n_keep = int(np.sum(S > cfg.orth_rank_tol * S[0]))
    Q = Q_full[:, :n_keep]
    if verbose:
        print(f'[orth] SVD: rank {n_keep} kept '
              f'(sigma[{n_keep-1}]/sigma[0]={S[n_keep-1]/S[0]:.2e})')

    # Fixed ANN inputs: [x_phys, x_aug=0, u] at the decimated samples (D7.4).
    nxd = cfg.nx_phys + cfg.nx_ann
    Z_pts = np.zeros((len(sub_ix), nxd + nu, 1), dtype=np.float64)
    Z_pts[:, :NX, 0] = X_flat[sub_ix]
    Z_pts[:, nxd:, 0] = U_flat[sub_ix]
    return Q, Z_pts, S


def build_orth_penalty(cfg: RunConfig, data, norm, verbose=True) -> OrthProjectionPenalty:
    """Load-or-build the penalty for this configuration (cached, D7.4)."""
    theta_bar = theta_bar_for(cfg)
    path = _cache_path(cfg, theta_bar)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        Q, Z_pts = z['Q'], z['Z_pts']
        if verbose:
            print(f'[orth] loaded cached penalty basis: {path} '
                  f'(Q {Q.shape}, {Z_pts.shape[0]} points)')
    else:
        Q, Z_pts, S = _compute(cfg, data, norm, theta_bar, verbose=verbose)
        np.savez_compressed(path, Q=Q, Z_pts=Z_pts, S=S, theta_bar=theta_bar,
                            config=json.dumps({'route': list(cfg.ann_route_ix),
                                               'stride': cfg.orth_point_stride}))
        if verbose:
            print(f'[orth] cached penalty basis: {path}')
    route_cols = [k for k, i in enumerate(cfg.ann_route_ix) if i < cfg.nx_phys]
    return OrthProjectionPenalty(Q, Z_pts, route_cols, cfg.orth_beta,
                                 dtype=cfg.dtype_pt)
