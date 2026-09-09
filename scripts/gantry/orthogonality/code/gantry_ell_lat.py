"""ell_lat versus ell_total at a TRAINED gantry checkpoint, through the shared component.

WHY. The testbed adapter reported, at an UNTRAINED network, that the latent route was nearly
orthogonal to the physical sensitivity space (ell 0.064) while the total contribution was
strongly aligned (ell 0.824). If that survives training, the alignment lives in the DIRECT
correction and the additional-state extension is not where the value is. The gantry says the
opposite at 90% latent share, so the two systems disagree and this run is the tiebreaker at a
converged model.

CHECKPOINT. Run 81758, the most converged of the thirteen saved runs in
`meeting-07-09-2026/server/` (best validation 5.7483e-06, nx_ann = 8, LBFGS-polished).
Every one of those runs has `joint = False` and `orth = False`, so this is the UNREGULARISED,
non-jointly-estimated network. It measures where the alignment sits when nothing opposes it.
It is not a test of the penalty.

WHAT IS SHARED WITH THE TESTBED. Geometry construction, encoder profiling, the ell / a_par /
a_perp split and the three-intervention identity all come from
`model_augmentation/fit_systems/`. Only the rollout differs, which is the point of the adapter
seam.

ONE CONSTRUCTION TO BE EXPLICIT ABOUT. The saved model's physical block is not parameterised
(the run had joint estimation off), so it cannot be differentiated with respect to theta. The
geometry therefore uses a PARAMETERISED copy of the physical block at the nominal parameters,
with the trained ANN and encoder unchanged. The two must agree numerically at nominal theta,
and that is asserted rather than assumed: if they disagree, the geometry describes a different
model from the contribution.

Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/gantry_ell_lat.py
"""
__project_origin__ = "added"

import os
import sys
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..', '..')))

from gantry_dynamic.config import RunConfig                      # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization  # noqa: E402
from gantry_dynamic.orth_penalty import theta_bar_for, _x_logical_from_data  # noqa: E402
from model_augmentation.fit_systems.blocks import Parameterized_Gantry_State_Block  # noqa: E402
from model_augmentation.fit_systems.trajectory_orth_projection import (  # noqa: E402
    TrajectoryProjectionGeometry)

F64 = torch.float64
OUT = os.path.join(HERE, 'gantry_ell_lat.json')
RUN = '81758'
CKPT = os.path.join(HERE, '..', '..', 'meeting', 'meeting-07-09-2026', 'server',
                    'augmentation_ma50_b140-230_a6_z03_linear_map', RUN, f'gantry_{RUN}')

NX_PHYS, NU, NY = 6, 3, 3
N_WIN, T_WIN = 24, 100


def cfg_for(run_cfg):
    return RunConfig(mode=run_cfg['MODE'], encoder_init=run_cfg['ENCODER_INIT'],
                     nx_ann=run_cfg['NX_ANN'], ann_route_ix=tuple(run_cfg['ANN_ROUTE_IX']),
                     n_nodes_per_layer=run_cfg['N_NODES_PER_LAYER'],
                     n_hidden_layers=run_cfg['N_HIDDEN_LAYERS'],
                     up_sample=run_cfg.get('UP_SAMPLE', 1), fs_new=run_cfg['FS_NEW'],
                     stride=run_cfg.get('STRIDE', 10), na_nb_override=run_cfg['NA_NB'],
                     nf_override=run_cfg['NF'], joint_estimation=False,
                     save_flag=False, device='cpu')


def main():
    torch.set_default_dtype(F64)
    print('=' * 78)
    print(f'GANTRY ell_lat vs ell_total.  Run {RUN}, trained, orth=OFF, joint=False.')
    print('=' * 78)

    rc = json.load(open(os.path.join(os.path.dirname(CKPT), 'config.json')))
    cfg = cfg_for(rc)
    print(f'  nx_ann={rc["NX_ANN"]}  route={len(rc["ANN_ROUTE_IX"])} rows  '
          f'fs={rc["FS_NEW"]}  nf={rc["NF"]}  joint={rc["JOINT_ESTIMATION"]}  '
          f'orth={rc["ORTH"]}')

    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    theta_bar = theta_bar_for(cfg)

    fs = torch.load(CKPT, map_location='cpu', weights_only=False)
    hfn = fs.hfn.to(F64)
    hfn.eval()
    nxd = fs.nx
    nx_ann = nxd - NX_PHYS
    ann = [b for b in hfn.connected_blocks if type(b).__name__ == 'Static_ANN_Block'][0]
    print(f'  loaded nx={nxd} (nx_ann={nx_ann})  bestfit={fs.bestfit:.4e}')

    # ---- parameterised physical block for the GEOMETRY only --------------------
    pblk = Parameterized_Gantry_State_Block(
        Y_op=None, std_x=norm.std_x, std_u=norm.std_u, x_mean=norm.x_mean,
        u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=cfg.up_sample,
        params_init=torch.tensor(theta_bar, dtype=F64)).to(F64)
    pblk.eval()
    # Detach `log_params` from the Parameter registry so it can be rebound to a dual tensor
    # during forward-mode differentiation (see the note in `roll`).
    _lp_nominal = pblk._parameters.pop('log_params').data.clone()
    pblk.log_params = _lp_nominal
    phys_blk = hfn.connected_blocks[0]
    Cd = hfn.connected_blocks[1]

    # ---- validation windows ----------------------------------------------------
    recs = []
    for sd in data.val_list:
        xl = _x_logical_from_data(sd)
        recs.append(((xl - norm.x_mean.flatten()) / norm.std_x.flatten(),
                     (sd.u - norm.u_mean.flatten()) / norm.std_u.flatten()))
    rng = np.random.default_rng(0)
    starts = [(rng.integers(len(recs)), rng.integers(1, recs[0][0].shape[0] - T_WIN - 1))
              for _ in range(N_WIN)]
    X0 = torch.tensor(np.stack([recs[r][0][k] for r, k in starts]), dtype=F64)
    Uw = torch.tensor(np.stack([recs[r][1][k:k + T_WIN] for r, k in starts]), dtype=F64)

    def phys_step(x, u, theta=None):
        z = torch.cat([x.unsqueeze(-1), u.unsqueeze(-1)], dim=1)
        if theta is None:
            return phys_blk.nonlinear_function(z).squeeze(-1)
        return pblk.nonlinear_function(z, log_params=theta).squeeze(-1) \
            if 'log_params' in pblk.nonlinear_function.__code__.co_varnames \
            else pblk.nonlinear_function(z).squeeze(-1)

    def roll(theta_log, mode):
        """mode: full | off | clamped. theta_log=None uses the saved (non-param) block."""
        x = X0.clone()
        xa = torch.zeros(N_WIN, nx_ann, dtype=F64)
        ys = []
        for k in range(T_WIN):
            u = Uw[:, k, :]
            if theta_log is None:
                xn_p = phys_blk.nonlinear_function(
                    torch.cat([x.unsqueeze(-1), u.unsqueeze(-1)], dim=1)).squeeze(-1)
            else:
                # Plain ATTRIBUTE assignment, not `.data`. Under forward-mode autodiff
                # `theta_log` is a dual tensor; writing it through `.data` strips the dual and
                # functorch raises "wrapper->level() <= current_level". `log_params` is popped
                # out of `_parameters` below so this binds a plain tensor and the dual
                # propagates. Nothing is mutated permanently: the nominal is restored after.
                pblk.log_params = theta_log
                xn_p = pblk.nonlinear_function(
                    torch.cat([x.unsqueeze(-1), u.unsqueeze(-1)], dim=1)).squeeze(-1)
            if mode == 'off':
                w = torch.zeros(N_WIN, nxd, dtype=F64)
            else:
                zin = torch.cat([x, xa, u], dim=1).unsqueeze(-1)
                w = ann(zin).squeeze(-1)
            x = xn_p + w[:, :NX_PHYS]
            xa = torch.zeros_like(xa) if mode in ('clamped', 'off') else w[:, NX_PHYS:]
            ys.append(x[:, :NY] if Cd is None else x @ torch.eye(NX_PHYS, dtype=F64)[:, :NY])
        return torch.stack(ys, dim=1).reshape(-1)

    # ---- parity: the parameterised copy must equal the saved block at nominal ----
    with torch.no_grad():
        a = roll(None, 'full')
        b = roll(_lp_nominal.clone(), 'full')
    rel = float((a - b).norm() / max(a.norm(), torch.tensor(1e-300)))
    print(f'\n  parity, saved block vs parameterised copy at nominal theta: {rel:.3e}')
    # Tolerance 1e-6, with the reason recorded rather than the threshold merely widened.
    # Measured 2026-09-07: the ONE-STEP difference is already 1.18e-08 and does NOT grow over
    # 100 steps (3.7e-8 -> 2.7e-8), so this is a constant offset, not accumulation or
    # instability. Source: the nominal parameter vector carries FLOAT32 rounding
    # (theta_bar[3] = 20.29999924) while the saved block uses the constants directly, and a
    # relative 1e-8 is float32-epsilon territory. The geometry is a DERIVATIVE with respect to
    # theta, so a nominal shifted by 1e-8 relative perturbs S by the same order and is
    # irrelevant. A gap that GREW with the horizon would not be, which is why it was checked.
    assert rel < 1e-6, ('the parameterised copy is NOT the saved model to float32 constant '
                        f'precision (rel {rel:.2e}); the geometry would describe a different '
                        'system from the contribution')

    # ---- geometry: S through the FULL augmented rollout --------------------------
    from torch.func import jacfwd
    lp = _lp_nominal.clone()
    S = jacfwd(lambda t: roll(t, 'full'))(lp).detach().numpy()
    S = S / theta_bar[None, :]                      # d/dlog theta -> d/dtheta
    geom = TrajectoryProjectionGeometry(S, E=None)  # encoder not in this rollout: x0 is data
    print('\n' + geom.summary())

    # ---- the three interventions ------------------------------------------------
    with torch.no_grad():
        full, off, clamped = roll(None, 'full'), roll(None, 'off'), roll(None, 'clamped')
    d_tot, d_lat, d_dir = full - off, full - clamped, clamped - off
    err = float((d_tot - d_lat - d_dir).abs().max()) / max(float(d_tot.abs().max()), 1e-300)
    print(f'\n  three-intervention identity residual: {err:.2e}')

    res = {'run': RUN, 'n_win': N_WIN, 't_win': T_WIN, 'rank_S': geom.rank_S,
           'parity': rel, 'identity_residual': err}
    print(f'\n  {"contribution":<10s} {"ell":>8s} {"a_par":>11s} {"a_perp":>11s} {"|d|":>11s}')
    for name, v in (('total', d_tot), ('latent', d_lat), ('direct', d_dir)):
        m = geom.contribution_metrics(v)
        res[name] = m
        print(f'  {name:<10s} {m["ell"]:8.4f} {m["a_par"]:11.3e} {m["a_perp"]:11.3e} '
              f'{m["norm_profiled"]:11.3e}')

    chance = np.sqrt(geom.rank_S / geom.N)
    print(f'\n  chance level sqrt({geom.rank_S}/{geom.N}) = {chance:.4f}')
    res['chance'] = float(chance)
    print('  Read ell against that, not against zero.')

    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f'\n  wrote {OUT}')


if __name__ == '__main__':
    main()
