"""Jan Hoekstra's plain black box (DECISIONS.md BB2-013) on the grey box's carrier.

Jan's model: `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`, deepSI 0.3.29 `SS_encoder_general_hf`
with f = `default_state_net`, h = `default_output_net` (2 x 8 tanh), encoder
`default_encoder_net` (2 x 16 tanh), y = h(x), x+ = f(x, u), no feedthrough, random init.
The only additions are the interface the grey box's carrier asks of an `hfn` (the same three
items as blackbox-closed-loop BB-003) and `.reshape` for `.view` (identical values).
"""
__project_origin__ = "added"

import numpy as np
import torch

from deepSI.fit_systems.encoders import (default_encoder_net, default_state_net,
                                         default_output_net, hf_net_default)

NX = 16                                               # BB2-013 HEURISTIC, search axis 6 to 32
F_KW = dict(n_hidden_layers=2, n_nodes_per_layer=8)   # Jan L27
H_KW = dict(n_hidden_layers=2, n_nodes_per_layer=8)   # Jan L27
E_KW = dict(n_hidden_layers=2, n_nodes_per_layer=16)  # Jan L28


class JanEncoder(default_encoder_net):
    def forward(self, upast, ypast):
        return self.net(torch.cat([upast.reshape(upast.shape[0], -1),
                                   ypast.reshape(ypast.shape[0], -1)], dim=1))


class JanStateNet(default_state_net):
    def forward(self, x, u):
        return self.net(torch.cat([x, u.reshape(u.shape[0], -1)], dim=1))


class JanHF(hf_net_default):
    """Jan's hf step. output_only(x) = h(x) (legal: no feedthrough); no physical block."""
    connected_blocks = ()

    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny, feedthrough=False, f_net=JanStateNet,
                         f_net_kwargs=dict(F_KW), h_net_kwargs=dict(H_KW),
                         h_net=default_output_net)

    def output_only(self, x, u=None):
        return self.hn(x)

    def init_model(self, sys_data=None):
        return None


def build(cfg, data, norm, seed, nx=NX, verbose=True):
    """Jan's black box on the grey box's carrier, data, norm, windows and closed loop."""
    from gantry_dynamic.controller import build_closed_loop
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed

    np.random.seed(seed)
    torch.manual_seed(seed)
    hf = JanHF(nx, cfg.nu, cfg.ny)
    na = nb = int(cfg.na_nb)
    fit_sys = SSE_Interconnect_Composed(interconnect=hf, na=na, nb=nb, na_right=1, nb_right=1,
                                        e_net=JanEncoder, e_net_kwargs=dict(E_KW))
    fit_sys.norm.u0 = norm.u_mean.flatten()
    fit_sys.norm.ustd = norm.std_u.flatten()
    fit_sys.norm.y0 = norm.y0
    fit_sys.norm.ystd = norm.ystd
    fit_sys.init_model(sys_data=data.train_data, device='cpu', auto_fit_norm=False,
                       optimizer_kwargs={'lr': cfg.lr, 'eps': cfg.adam_eps})
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(cfg.dtype_pt)
    fit_sys.burn_in = cfg.burn_in
    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg, train_files=data.train_files, val_files=data.val_files,
        val_data=data.val_ckpt_data, verbose=verbose)
    if verbose:
        n_par = sum(p.numel() for m in (fit_sys.encoder, fit_sys.hfn) for p in m.parameters())
        print('[jan] seed %d nx %d na=nb=%d params %d lr %g' % (seed, nx, na, n_par, cfg.lr),
              flush=True)
    return fit_sys
