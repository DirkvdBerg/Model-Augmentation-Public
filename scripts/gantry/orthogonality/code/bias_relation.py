"""The bias relation: what unmodelled dynamics does to the physical parameters.

Supersedes the floor computation in `recovery_floor.py`, which violated RULES.md Rule 1:
`||Q^T Delta_true|| / ||Delta_true||` needs the 8-state truth, so it cannot be computed on
Telica and is not a deliverable.

THE DELIVERABLE (category 1, structural). Under any orthogonality scheme the in-span part
of the model mismatch is credited to theta, which the by-construction paper writes as its
Eq. 8. In our coordinates:

    theta_err = (Phi_k^T Phi_k)^-1 Phi_k^T Delta          [identity]
    ||theta_err|| <= ||Delta|| / sigma_min(Phi_k)          [bound]

This says what determines the bias for ANY unmodelled dynamics, not what the bias happens to
be for this absorber. Both parts of the bound are computable without truth.

CATEGORY 2, and this is what ships to Telica:
  sigma_min(Phi_k)  from the model and the excitation. Note it must be evaluated in the
                    IDENTIFIABLE coordinates kappa: on the raw 14 parameters sigma_min = 0,
                    the pseudo-inverse does not exist and the bound is vacuous. This is a
                    third independent reason for the reparameterisation, after Assumption 1
                    of the by-construction paper and the removal of the flat directions.
  Delta_obs         the one-step transition residual on RECONSTRUCTED states,
                        Delta_obs[k] = x_hat[k+1] - f_base(x_hat[k], u[k]),
                    with x_hat = [P^-T y, FD velocities] exactly as D-111 builds the states
                    for Phi. Available on any dataset, real or simulated.
  theta_err_obs     the induced parameter error, in physical units and as a fraction of each
                    combination, which is the number a committee actually asks for.

CATEGORY 3, validation only, reported as such and never on its own:
  Delta_true        the 8-state truth minus the baseline at theta_true.
  Checks: (a) is Delta_obs a good proxy for Delta_true, (b) does the bound hold, (c) is it
  vacuous. If Delta_obs tracks Delta_true here, the category-2 pipeline is trustworthy where
  no truth exists.

Phi_k is obtained from Phi_theta without a second Jacobian pass. Since the baseline factors
through kappa (proved in `claimA_symbolic.py` and `.m`), Phi_theta = Phi_k J_k with
J_k = d kappa / d theta of full row rank 10, so Phi_k = Phi_theta J_k^+ and
Phi_k J_k = Phi_theta exactly. That identity is asserted numerically below.

Outputs -> scripts/gantry/orthogonality/code/bias_relation.json
Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
    -n GraduationProject python -u scripts/gantry/orthogonality/code/bias_relation.py
"""
import os
import sys
import json
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import load_datasets, compute_normalization, TRAIN_FILES, load_mat_aug
from gantry_dynamic import oracle
from gantry_dynamic.orth_penalty import _x_logical_from_data, _make_block, theta_bar_for
from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block
from model_augmentation.systems.gantry_ss import P as _P

_F64 = torch.float64
OUT = os.path.join(_HERE, 'bias_relation.json')
P_np = _P.numpy().astype(np.float64)
CFG = RunConfig(use_f64=True, snr=None, up_sample=2)

PNAMES = list(Parameterized_Gantry_State_Block.PARAM_NAMES)
IX = {n: i for i, n in enumerate(PNAMES)}
KNAMES = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
          "m_total", "m_diff", "J_eff", "d"]


def kappa_jacobian(Lb):
    """J_k = d kappa / d theta, 10 x 14, from blocks.py::_combos_from_raw."""
    J = np.zeros((10, 14))
    J[0, IX['kb1']] = J[0, IX['kb2']] = 1.0
    J[1, IX['cg1']] = 1.0
    J[2, IX['cg2']] = 1.0
    J[3, IX['cy']] = 1.0
    J[4, IX['cb1']] = J[4, IX['cb2']] = 1.0
    J[5, IX['mh']] = 1.0
    J[6, IX['m1']] = J[6, IX['m2']] = J[6, IX['mb']] = 1.0
    J[7, IX['m1']], J[7, IX['m2']] = 1.0, -1.0
    J[8, IX['Jb']] = J[8, IX['Jh']] = 1.0
    J[8, IX['m1']] = J[8, IX['m2']] = Lb ** 2 / 4
    J[9, IX['d']] = 1.0
    return J


def kappa_of(theta, Lb):
    t = {n: theta[IX[n]] for n in PNAMES}
    return np.array([t['kb1'] + t['kb2'], t['cg1'], t['cg2'], t['cy'],
                     t['cb1'] + t['cb2'], t['mh'],
                     t['m1'] + t['m2'] + t['mb'], t['m1'] - t['m2'],
                     t['Jb'] + t['Jh'] + (t['m1'] + t['m2']) * Lb ** 2 / 4,
                     t['d']])


def truth_step(x6_phys, a2, u_stage, ts, up_sample):
    """One RK4 step of the 8-state truth -> next 6 physical logical states."""
    x = np.array([x6_phys[0], x6_phys[1], x6_phys[2], a2[0],
                  x6_phys[3], x6_phys[4], x6_phys[5], a2[1]], dtype=np.float64)
    u_log = P_np @ np.asarray(u_stage, dtype=np.float64)
    h = ts / up_sample
    for _ in range(up_sample):
        k1 = oracle._deriv(x, u_log)
        k2 = oracle._deriv(x + 0.5 * h * k1, u_log)
        k3 = oracle._deriv(x + 0.5 * h * k2, u_log)
        k4 = oracle._deriv(x + h * k3, u_log)
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return np.array([x[0], x[1], x[2], x[4], x[5], x[6]])


def main():
    print('loading data...', flush=True)
    data = load_datasets(CFG)
    norm = compute_normalization(CFG, data)
    K0, NX = CFG.na_nb, CFG.nx_phys
    r_phys = [i for i in CFG.ann_route_ix if i < NX]
    n_r = len(r_phys)
    sx, mx = norm.std_x.flatten(), norm.x_mean.flatten()
    su, mu = norm.std_u.flatten(), norm.u_mean.flatten()

    theta = np.asarray(theta_bar_for(CFG), dtype=np.float64)
    blk = _make_block(CFG, norm, theta)
    Lb = float(blk.Lb.item())
    print(f'theta_bar (deployed linearisation point) = '
          f'{np.array2string(theta, precision=4)}')
    print(f'K0={K0}, rows={r_phys}, stride={CFG.orth_point_stride}, Lb={Lb}')

    # ---- stacks, in the deployed construction (D-111 reconstructed states) ----
    Xp_parts, Us_parts, A_parts, Xp_next_parts = [], [], [], []
    for f, sd in zip(TRAIN_FILES, data.train_list):
        u_m, _, _, xaug_m = load_mat_aug(f, CFG)
        n = len(sd.u)
        assert np.allclose(u_m[:n], sd.u, atol=1e-9), f'u mismatch on {f}'
        xh = _x_logical_from_data(sd).astype(np.float64)      # (N,6) reconstructed
        # drop the last sample of each record: x_hat[k+1] must exist for Delta_obs
        Xp_parts.append(xh[K0:-1])
        Xp_next_parts.append(xh[K0 + 1:])
        Us_parts.append(sd.u[K0:-1].astype(np.float64))
        A_parts.append(xaug_m[:n][K0:-1].astype(np.float64))
    Xp = np.concatenate(Xp_parts)
    Xp_next = np.concatenate(Xp_next_parts)
    Us = np.concatenate(Us_parts)
    Aa = np.concatenate(A_parts)
    X_flat = (Xp - mx) / sx
    U_flat = (Us - mu) / su
    sub_ix = np.arange(0, X_flat.shape[0], CFG.orth_point_stride)
    print(f'samples {X_flat.shape[0]}, decimated points {len(sub_ix)}')

    # ---- one pass: Phi_theta, Delta_obs (cat 2), Delta_true (cat 3) ----
    n_rows = len(sub_ix) * n_r
    Phi = np.empty((n_rows, 14))
    D_obs = np.empty(n_rows)
    D_true = np.empty(n_rows)
    theta_t = torch.tensor(theta, dtype=_F64)
    ts, ups = CFG.ts_new, CFG.up_sample
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
        Jr = (J / theta_t.view(1, 14)).detach()[r_phys, :]
        Phi[si * n_r:(si + 1) * n_r, :] = Jr.numpy()

        xb = xn.detach().numpy()                                  # baseline next (norm)
        xo = (Xp_next[ix] - mx) / sx                              # observed next (norm)
        D_obs[si * n_r:(si + 1) * n_r] = (xo - xb)[r_phys]
        xt = (truth_step(Xp[ix], Aa[ix], Us[ix], ts, ups) - mx) / sx
        D_true[si * n_r:(si + 1) * n_r] = (xt - xb)[r_phys]
        if (si + 1) % 1000 == 0:
            print(f'  {si+1}/{len(sub_ix)} ({(si+1)/(time.time()-t0):.1f}/s)', flush=True)

    # ---- Phi in identifiable coordinates ----
    Jk = kappa_jacobian(Lb)
    Phi_k = Phi @ np.linalg.pinv(Jk)
    rec = np.linalg.norm(Phi_k @ Jk - Phi) / np.linalg.norm(Phi)
    print(f'\n[check] ||Phi_k Jk - Phi|| / ||Phi|| = {rec:.3e}  '
          f'(claim A: must be ~0)')

    s_th = np.linalg.svd(Phi, compute_uv=False)
    s_k = np.linalg.svd(Phi_k, compute_uv=False)
    print(f'sigma_min(Phi_theta) = {s_th[-1]:.3e}  (rank-deficient, bound vacuous)')
    print(f'sigma_min(Phi_kappa) = {s_k[-1]:.3e}   cond = {s_k[0]/s_k[-1]:.3e}')

    kap = kappa_of(theta, Lb)
    Pk_pinv = np.linalg.pinv(Phi_k)

    res = {'category_note': 'see documentation/RULES.md Rule 1',
           'n_points': int(len(sub_ix)), 'n_r': n_r,
           'theta_bar': theta.tolist(), 'kappa_names': KNAMES,
           'kappa_at_theta_bar': kap.tolist(),
           'phi_k_identity_residual': float(rec),
           'sigma_min_Phi_theta': float(s_th[-1]),
           'sigma_min_Phi_kappa': float(s_k[-1]),
           'cond_Phi_kappa': float(s_k[0] / s_k[-1])}

    for tag, D in [('obs', D_obs), ('true', D_true)]:
        te = Pk_pinv @ D
        nD = float(np.linalg.norm(D))
        bound = nD / s_k[-1]
        rel = te / kap
        print(f'\n--- Delta_{tag}: ||Delta|| = {nD:.4e} ---')
        print(f'  ||theta_err|| = {np.linalg.norm(te):.4e}   '
              f'bound ||Delta||/sigma_min = {bound:.4e}   '
              f'slack = {bound/np.linalg.norm(te):.1f}x')
        print('  induced error per identifiable combination:')
        for i, n in enumerate(KNAMES):
            print(f'    {n:9s} {te[i]:+.4e}  ({100*rel[i]:+.3f} % of {kap[i]:.4g})')
        res[f'norm_Delta_{tag}'] = nD
        res[f'theta_err_{tag}'] = te.tolist()
        res[f'theta_err_rel_{tag}'] = rel.tolist()
        res[f'bound_{tag}'] = float(bound)
        res[f'bound_slack_{tag}'] = float(bound / np.linalg.norm(te))

    # validation: is the data-computable proxy faithful to the truth?
    cos = float(D_obs @ D_true / (np.linalg.norm(D_obs) * np.linalg.norm(D_true)))
    print(f'\n[validation, category 3] cos(Delta_obs, Delta_true) = {cos:.4f}, '
          f'||D_obs||/||D_true|| = {np.linalg.norm(D_obs)/np.linalg.norm(D_true):.4f}')
    res['cos_Dobs_Dtrue'] = cos
    res['norm_ratio_Dobs_Dtrue'] = float(np.linalg.norm(D_obs) / np.linalg.norm(D_true))

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
