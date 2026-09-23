"""Model construction for real Telica data (G6): gantry_dynamic/model.py::build_model with the
physical block replaced by the G4 baseline (attempt-2 parameters + tanh friction, TR-015).

Everything else is the vendored structure unchanged: Interconnect, linear output block, static
ANN block routed additively into the state update, SSE_Interconnect_Composed, linear-map encoder
(Hoekstra 2026 Eq. 16-17) from the linearised gantry at the pipeline rate.
"""
import json
import os

import numpy as np
import torch
import deepSI

from model_augmentation.utils.utils import (expansion_matrix, selection_matrix,
                                            normalize_linear_ss_matrices)
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect_Composed
from model_augmentation.fit_systems.blocks import Linear_Output_Block, Static_ANN_Block
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

from friction.karnopp import GantryFrictionBlock, rebuild_float64

BASELINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'baseline')
PHY_IX = np.arange(6)


def baseline_params(tag):
    """(raw14 dict, cc tuple) of the chosen G4 baseline ('' = attempt 1, '_a2' = attempt 2)."""
    R = json.load(open(os.path.join(BASELINE_DIR, f'recovered_params{tag}.json')))
    return R['raw14'], tuple(R['cc'])


def _synthetic_sysdata(norm):
    """deepSI System_data whose population std/mean equal the FROZEN normalisation exactly
    (two rows m +- s): the encoder scaling needs std only, never the training data (TR-016)."""
    def pm(mean, std):
        m, s = np.ravel(mean), np.ravel(std)
        return np.stack([m + s, m - s])
    u = pm(norm.u_mean, norm.std_u); y = pm(norm.y0, norm.ystd); x = pm(norm.x_mean, norm.std_x)
    sd = deepSI.System_data(u=u, y=y)
    sd.x = x
    return sd


def build_model_real(hp, cfg, data, norm, baseline_tag='_a2', fric_mode='tanh'):
    NX_PHYS = cfg.nx_phys
    nu, ny = cfg.nu, cfg.ny
    DT = cfg.dtype_pt
    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN
    route_ix = np.asarray(cfg.ann_route_ix)
    na = hp.get('na_nb', 2 * nxd + 1); nb = na
    na_right = nb_right = 1 if cfg.encoder_init == 'linear_map' else 0

    ic = Interconnect(nxd, nu, ny, debugging=False)
    raw, cc = baseline_params(baseline_tag)
    phy = GantryFrictionBlock(cc=cc, mode=fric_mode, Y_op=None, std_x=norm.std_x,
                              std_u=norm.std_u, x_mean=norm.x_mean, u_mean=norm.u_mean,
                              Ts=cfg.ts_new, up_sample=hp['up_sample'])
    phy = rebuild_float64(phy, p=raw)
    phy.cc_init = torch.as_tensor(cc, dtype=torch.float64).reshape(1, 3).clone()
    # the block stored its normalisation via to_tensor (float32); restore the float64 values so
    # the block, the encoder and fit_sys.norm share one set of constants (TR-013)
    f64 = lambda a, shp: torch.as_tensor(np.asarray(a, np.float64)).reshape(shp)   # noqa: E731
    phy.std_x = f64(norm.std_x, (6, 1)); phy.std_x_1d = f64(norm.std_x, (6,))
    phy.std_u = f64(norm.std_u, (3, 1)); phy.x_mean = f64(norm.x_mean, (6, 1))
    phy.u_mean = f64(norm.u_mean, (3, 1))
    phy.early_exit = False             # fixed 4 active-set passes: compile-safe, same result
    phy = phy.to(DT)
    out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)
    ic.add_block(phy)
    ic.add_block(out_phys)
    _act = torch.nn.Identity if cfg.ann_activation == 'linear' else torch.nn.Tanh
    ann = Static_ANN_Block(nz=nxd + nu, nw=len(route_ix),
                           n_nodes_per_layer=hp['n_nodes_per_layer'],
                           n_hidden_layers=hp['n_hidden_layers'],
                           net=zero_init_feed_forward_nn, activation=_act)
    ic.add_block(ann)
    ic.connect_block_signals(ann, ["x", "u"], [])
    ic.connect_signals(ann, "xp", "additive", expansion_matrix(route_ix, nxd))
    ic.connect_signals("x", phy, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy, ["u"], [])
    ic.connect_signals(phy, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])

    fit_sys = SSE_Interconnect_Composed(
        interconnect=ic, na=na, nb=nb, na_right=na_right, nb_right=nb_right,
        e_net_kwargs={"n_nodes_per_layer": hp['n_nodes_per_layer'],
                      "n_hidden_layers": hp['n_hidden_layers']})
    fit_sys.norm.u0 = np.ravel(norm.u_mean)
    fit_sys.norm.ustd = np.ravel(norm.std_u)
    fit_sys.norm.y0 = np.ravel(norm.y0)
    fit_sys.norm.ystd = np.ravel(norm.ystd)
    if cfg.encoder_init == 'linear_map':
        Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=cfg.ts_new)
        Ab, Bb, Cb, Db = normalize_linear_ss_matrices(Ad, Bd, Cd_dt, Dd_dt, _synthetic_sysdata(norm))
        fit_sys.encoder = linear_encoder_init_aug(
            A=Ab, B=Bb, C=Cb, D=Db, nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb, nx_aug=NX_ANN,
            n_nodes_per_layer=hp['n_nodes_per_layer'], n_hidden_layers=hp['n_hidden_layers'],
            flag_linear_only=False, u_mean=norm.u_mean, std_u=norm.std_u, y0=norm.y0,
            ystd=norm.ystd, x_mean=norm.x_mean, std_x=norm.std_x, dtype=DT).to(DT)
    fit_sys.init_model(sys_data=data.train_data, device='cpu', auto_fit_norm=False,
                       optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps})
    fit_sys.hfn.to(DT)
    fit_sys.burn_in = cfg.burn_in
    return fit_sys
