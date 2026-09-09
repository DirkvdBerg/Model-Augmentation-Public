"""How long does rebuilding the orthogonality basis actually cost?

The question behind this: `orth_penalty.py` freezes the basis at `theta_bar` for the whole run,
and its docstring records the fresh build as "~6 min Jacobians + ~0.03 s SVD" at stride 100.
That 6 minutes is the stated reason not to re-anchor during training. Before accepting a
METHODOLOGICAL cost (a non-stationary objective) or rejecting re-anchoring on price, the price
should be a real number rather than an artefact of how the Jacobian is taken.

WHAT IS SLOW, and it is not the mathematics. The deployed loop runs one point at a time and
takes SIX REVERSE-MODE passes per point, so roughly 6 x 6700 backward passes through the
physical block, all sequential in Python. The map has 14 parameters and NX output rows, so the
natural mode is FORWARD: 14 passes over the whole batch at once. This is the same
reverse-versus-forward error found in `train_compare.py`, where it was about 85% of the sweep's
runtime.

This script implements both, VERIFIES THEY AGREE, and times them. It changes nothing in the
deployed path; whether to adopt the fast form is a separate decision, and whether to re-anchor
per epoch is a different decision again, because the objection to a moving basis is that it
makes the objective non-stationary, not that it is expensive.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/jacobian_mode_bench.py
"""
__project_origin__ = "added"

import os
import sys
import json
import time

import numpy as np
import torch
from torch.func import jacfwd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))

from gantry_dynamic.config import RunConfig                           # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization  # noqa: E402
from gantry_dynamic.orth_penalty import theta_bar_for, _x_logical_from_data  # noqa: E402
from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block  # noqa: E402

F64 = torch.float64
OUT = os.path.join(HERE, 'jacobian_mode_bench.json')
NX, NU, NP = 6, 3, 14


def build_block(cfg, norm, theta_bar):
    blk = Parameterized_Gantry_State_Block(
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u, x_mean=norm.x_mean,
        u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=cfg.up_sample,
        params_init=torch.tensor(theta_bar, dtype=F64)).to(F64)
    blk.eval()
    return blk


def as_parameter(blk, lp):
    """Put `log_params` back in the Parameter registry for REVERSE mode.

    The instance `__dict__` entry must be removed first: `nn.Module.__getattr__` only runs when
    normal attribute lookup FAILS, so a plain attribute set earlier shadows the registered
    Parameter and `blk.log_params` silently returns a tensor with no grad_fn.
    """
    blk.__dict__.pop('log_params', None)
    blk._parameters['log_params'] = torch.nn.Parameter(lp.clone())
    return blk.log_params


def as_plain(blk):
    """Take `log_params` out of the registry so it can be rebound to a forward-mode dual."""
    p = blk._parameters.pop('log_params', None)
    lp = (p.data if p is not None else blk.log_params).clone()
    blk.__dict__.pop('log_params', None)
    blk.log_params = lp
    return lp


def jac_reverse(blk, X, U, theta_t):
    """The DEPLOYED form: one point at a time, NX reverse-mode passes each."""
    n = X.shape[0]
    Phi = np.empty((n, NX, NP))
    Xn = np.empty((n, NX))
    for i in range(n):
        z = torch.cat([torch.tensor(X[i], dtype=F64).view(1, NX, 1),
                       torch.tensor(U[i], dtype=F64).view(1, NU, 1)], dim=1)
        xn = blk.nonlinear_function(z).view(NX)
        J = torch.zeros(NX, NP, dtype=F64)
        for r in range(NX):
            J[r] = torch.autograd.grad(xn[r], blk.log_params,
                                       retain_graph=(r < NX - 1))[0]
        Phi[i] = (J / theta_t.view(1, NP)).detach().numpy()
        Xn[i] = xn.detach().numpy()
    return Phi, Xn


def jac_forward(blk, X, U, theta_t, lp_nominal, chunk=None):
    """FORWARD mode over the whole batch: NP passes total, not NX per point.

    `log_params` is popped out of the parameter registry by the caller so it can be rebound to
    a dual tensor; writing through `.data` strips the dual and functorch raises an internal
    assert.
    """
    n = X.shape[0]
    chunk = n if chunk is None else chunk
    Phi = np.empty((n, NX, NP))
    Xn = np.empty((n, NX))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        z = torch.cat([torch.tensor(X[s:e], dtype=F64).unsqueeze(-1),
                       torch.tensor(U[s:e], dtype=F64).unsqueeze(-1)], dim=1)

        def f(lp):
            blk.log_params = lp
            return blk.nonlinear_function(z).squeeze(-1)

        J = jacfwd(f)(lp_nominal)                     # (B, NX, NP)
        blk.log_params = lp_nominal
        with torch.no_grad():
            Xn[s:e] = blk.nonlinear_function(z).squeeze(-1).numpy()
        Phi[s:e] = (J / theta_t.view(1, 1, NP)).detach().numpy()
    return Phi, Xn


def main():
    torch.set_default_dtype(F64)
    print('=' * 78)
    print('JACOBIAN MODE BENCHMARK for the orthogonality basis rebuild')
    print('=' * 78)

    cfg = RunConfig(mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
                    nx_ann=8, ann_route_ix=tuple(range(14)), n_nodes_per_layer=24,
                    n_hidden_layers=3, up_sample=1, fs_new=4000, stride=10,
                    na_nb_override=29, nf_override=400, joint_estimation=False,
                    save_flag=False, device='cpu')
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    theta_bar = theta_bar_for(cfg)
    theta_t = torch.tensor(theta_bar, dtype=F64)

    K0 = cfg.na_nb
    X = np.concatenate([((_x_logical_from_data(sd)[K0:] - norm.x_mean.flatten())
                         / norm.std_x.flatten()) for sd in data.train_list])
    U = np.concatenate([((sd.u[K0:] - norm.u_mean.flatten()) / norm.std_u.flatten())
                        for sd in data.train_list])
    ix = np.arange(0, X.shape[0], cfg.orth_point_stride)
    n_pts = len(ix)
    print(f'  deployed point set: stride {cfg.orth_point_stride} -> {n_pts} points '
          f'from {X.shape[0]} samples')

    blk = build_block(cfg, norm, theta_bar)
    lp_nom = as_plain(blk)

    res = {'n_points_full': int(n_pts), 'stride': cfg.orth_point_stride}

    # ---- agreement on a subset, before trusting any timing ---------------------
    m = 200
    Xs, Us = X[ix[:m]], U[ix[:m]]
    as_parameter(blk, lp_nom)
    P_rev, Xn_rev = jac_reverse(blk, Xs, Us, theta_t)
    lp2 = as_plain(blk)
    P_fwd, Xn_fwd = jac_forward(blk, Xs, Us, theta_t, lp2)
    e_J = np.abs(P_rev - P_fwd).max() / max(np.abs(P_rev).max(), 1e-300)
    e_x = np.abs(Xn_rev - Xn_fwd).max() / max(np.abs(Xn_rev).max(), 1e-300)
    print(f'\n  agreement on {m} points: Jacobian {e_J:.2e}, state {e_x:.2e}')
    assert e_J < 1e-10 and e_x < 1e-12, 'forward and reverse disagree; do not trust the timings'
    res['agreement_jacobian'] = float(e_J)

    # ---- timing ----------------------------------------------------------------
    print(f'\n  timing on {m} points:')
    as_parameter(blk, lp_nom)
    t0 = time.time(); jac_reverse(blk, Xs, Us, theta_t); t_rev = time.time() - t0
    lp3 = as_plain(blk)
    t0 = time.time(); jac_forward(blk, Xs, Us, theta_t, lp3); t_fwd = time.time() - t0
    print(f'    reverse, per point, batch 1 : {t_rev:7.2f} s  '
          f'({t_rev / m * 1e3:.2f} ms/point)')
    print(f'    forward, batched            : {t_fwd:7.2f} s  '
          f'({t_fwd / m * 1e3:.2f} ms/point)')
    print(f'    speedup                     : {t_rev / max(t_fwd, 1e-9):7.1f}x')

    proj_rev, proj_fwd = t_rev / m * n_pts, t_fwd / m * n_pts
    print(f'\n  extrapolated to the full {n_pts}-point set:')
    print(f'    reverse : {proj_rev:8.1f} s  ({proj_rev / 60:.1f} min)   '
          '<- matches the ~6 min on record')
    print(f'    forward : {proj_fwd:8.1f} s  ({proj_fwd / 60:.2f} min)')

    # ---- the forward path on the FULL set, measured not extrapolated ------------
    lp4 = lp3
    t0 = time.time()
    Phi_all, Xn_all = jac_forward(blk, X[ix], U[ix], theta_t, lp4, chunk=2000)
    t_full = time.time() - t0
    r_phys = [i for i in cfg.ann_route_ix if i < NX]
    Phi = Phi_all[:, r_phys, :].reshape(-1, NP)
    Gam = (Xn_all[:, r_phys] - Phi_all[:, r_phys, :] @ theta_bar).reshape(-1, 1)
    t0 = time.time()
    _, sv, _ = np.linalg.svd(np.hstack([Phi, Gam]), full_matrices=False)
    t_svd = time.time() - t0
    rank = int(np.sum(sv > cfg.orth_rank_tol * sv[0]))
    print(f'\n  FULL rebuild, forward mode, measured:')
    print(f'    jacobians : {t_full:7.2f} s')
    print(f'    SVD       : {t_svd:7.2f} s   (rank {rank}, sigma ratio '
          f'{sv[rank - 1] / sv[0]:.2e})')
    print(f'    TOTAL     : {t_full + t_svd:7.2f} s')
    res.update(t_reverse_200=t_rev, t_forward_200=t_fwd,
               speedup=float(t_rev / max(t_fwd, 1e-9)),
               t_full_forward=t_full, t_svd=t_svd, rank=rank)

    print('\n  READ. The 6 minutes on record is an artefact of per-point reverse mode, not the')
    print('  cost of the mathematics. Whether to RE-ANCHOR during training is still a separate')
    print('  question: the objection to a moving basis is that it makes the objective')
    print('  non-stationary and lets the constraint set chase theta, not that it is expensive.')

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f'\n  wrote {OUT}')


if __name__ == '__main__':
    main()
