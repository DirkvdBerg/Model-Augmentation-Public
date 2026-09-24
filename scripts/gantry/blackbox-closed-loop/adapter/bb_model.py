"""The full black box as the grey box's own fit system with a black-box `hfn` (BB-003, BB-004).

Jan's ANN-SS (`scripts/gantry/msd_ndof_deepSI_encoder.py`, deepSI 0.3.29 `SS_encoder_general_hf`):
encoder, f and h are all `simple_res_net` MLPs, y = h(x), x+ = f(x, u), no feedthrough. What is
new here is only the carrier: the nets sit inside `SSE_Interconnect_Composed`, so the closed-loop
loss, the burn-in slice, fit(), validation and selection are the grey box's code, not a copy.
"""
import os

import numpy as np
import torch
from torch import nn

from deepSI.fit_systems.encoders import (default_encoder_net, default_state_net,
                                         default_output_net, hf_net_default)

# BB-004 settings. Sources in DECISIONS.md.
NX = 8                                   # Jan L30: nx = 2*dof, dof = 4 (X, Theta, Y, absorber)
F_KW = dict(n_hidden_layers=2, n_nodes_per_layer=8)       # Jan L27 (f and h)
H_KW = dict(n_hidden_layers=2, n_nodes_per_layer=8)       # Jan L27
E_KW = dict(n_hidden_layers=2, n_nodes_per_layer=16)      # Jan L28
JAN_LR = 1e-3                                              # Jan L33: deepSI's default Adam (G3, G4 smoke)
# BB-013: measured margin delta* = 1e-6 (outputs/g4/lr_margin.json), lr = delta*/10. Jan's 1e-3
# destabilised the closed loop in ONE step (run g4_smoke).
BB_LR, BB_EPS = 1e-7, 1e-8
NA_RIGHT = 1                                               # grey box get_encoder_dims, linear_map
ARMS = ('random', 'n4sid', 'frf')
BLA_RECORD = 'T10_aprbs_60.mat'                            # BB-008, the existing N4SID arm's record
FRF_CT = os.path.join('scripts', 'gantry', 'ann-blackbox', 'results',
                      'frf_init_joint_lowf_ma50_a5', 'frf_init_ss.npz')


class ReshapeEncoder(default_encoder_net):
    """deepSI's encoder with .reshape for .view (identical values; full-blackbox README step 1)."""

    def forward(self, upast, ypast):
        return self.net(torch.cat([upast.reshape(upast.shape[0], -1),
                                   ypast.reshape(ypast.shape[0], -1)], dim=1))


class ReshapeStateNet(default_state_net):
    def forward(self, x, u):
        return self.net(torch.cat([x, u.reshape(u.shape[0], -1)], dim=1))


class BlackBoxHF(hf_net_default):
    """Jan's hf step plus the three things the grey-box carrier asks of an `hfn`.

    output_only(x) = h(x): the closed-loop step needs y before u (Dc != 0), legal because D = 0.
    connected_blocks = (): no physical block, so no parameter prior, orth term or OBC.
    init_model: no-op (SSE_Interconnect.init_model calls it).
    """
    connected_blocks = ()

    def __init__(self, nx, nu, ny):
        super().__init__(nx, nu, ny, feedthrough=False, f_net=ReshapeStateNet,
                         f_net_kwargs=dict(F_KW), h_net_kwargs=dict(H_KW),
                         h_net=default_output_net)

    def output_only(self, x, u=None):
        return self.hn(x)

    def init_model(self, sys_data=None):
        return None


def build_blackbox(cfg, data, norm, arm='random', attach=True, verbose=True, train_files=None):
    """Black box on the grey box's data, norm, windows and closed loop. Returns fit_sys.

    cfg is the grey box's CFG (locally with device/compile_mode overridden, BB-001). The optimizer
    is built with the BB-004 values; everything else comes from cfg. `train_files` defaults to the
    grey box's TRAIN_FILES and must list exactly the records in `data.train_data`, in order (the
    G4 smoke subset, BB-011, is the only caller that passes it).
    """
    from gantry_dynamic.data import TRAIN_FILES, VAL_FILES
    from gantry_dynamic.controller import build_closed_loop
    from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed

    if arm not in ARMS:
        raise ValueError('arm must be one of %s, got %r' % (ARMS, arm))
    na = nb = int(cfg.na_nb)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    hf = BlackBoxHF(NX, cfg.nu, cfg.ny)
    fit_sys = SSE_Interconnect_Composed(interconnect=hf, na=na, nb=nb,
                                        na_right=NA_RIGHT, nb_right=NA_RIGHT,
                                        e_net=ReshapeEncoder, e_net_kwargs=dict(E_KW))
    # Manual normalisation, exactly the grey box's (model.py build_model), auto_fit_norm=False.
    fit_sys.norm.u0 = norm.u_mean.flatten()
    fit_sys.norm.ustd = norm.std_u.flatten()
    fit_sys.norm.y0 = norm.y0
    fit_sys.norm.ystd = norm.ystd
    fit_sys.init_model(sys_data=data.train_data, device='cpu', auto_fit_norm=False,
                       optimizer_kwargs={'lr': BB_LR, 'eps': BB_EPS})
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(cfg.dtype_pt)
    fit_sys.burn_in = cfg.burn_in                      # D-178 slice lives in Composed.loss

    if arm != 'random':
        apply_init(fit_sys, cfg, arm, verbose=verbose)
    if attach:
        fit_sys.simulator = build_closed_loop(
            fit_sys, norm, cfg, train_files=TRAIN_FILES if train_files is None else train_files,
            val_files=VAL_FILES,
            val_data=data.val_ckpt_data, verbose=verbose)
    if verbose:
        n_par = sum(p.numel() for m in (fit_sys.encoder, fit_sys.hfn) for p in m.parameters())
        print('[bb] arm=%s nx=%d na=nb=%d na_right=nb_right=%d params=%d lr=%g eps=%g burn_in=%d'
              % (arm, NX, na, NA_RIGHT, n_par, BB_LR, BB_EPS, fit_sys.burn_in))
    return fit_sys


def c2d_zoh(A, B, Ts):
    # THEORY: ZOH discretisation via the augmented matrix exponential (frf_init.py `c2d`).
    from scipy.linalg import expm
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n], M[:n, n:] = A, B
    E = expm(M * Ts)
    return E[:n, :n], E[:n, n:]


def apply_init(fit_sys, cfg, arm, verbose=True):
    """BB-008 initialisations, through the vendored bla_init functions (unchanged)."""
    from gantry_dynamic.data import load_traj
    from bbcl import REPO, OUT
    import bla_init
    rec = load_traj(BLA_RECORD, cfg)
    if arm == 'n4sid':
        bla_init.apply_bla_init(fit_sys, rec, mode='dyn', zero_nonlinear=True, verbose=verbose)
    elif arm == 'frf':
        d = np.load(os.path.join(REPO, FRF_CT))
        Ad, Bd = c2d_zoh(np.asarray(d['Ac'], float), np.asarray(d['Bc'], float), cfg.ts_new)
        os.makedirs(os.path.join(OUT, 'init'), exist_ok=True)
        path = os.path.join(OUT, 'init', 'frf_ss_fs%d.npz' % round(1 / cfg.ts_new))
        np.savez(path, A=Ad, B=Bd, C=np.asarray(d['C'], float), fs=1 / cfg.ts_new,
                 source=FRF_CT)
        if verbose:
            print('[bb] FRF CT model %s discretised ZOH at %g s -> %s' % (FRF_CT, cfg.ts_new, path))
        bla_init.apply_frf_bla_init(fit_sys, rec, ss_path=path, zero_nonlinear=True,
                                    verbose=verbose)
    # bla_init writes float32 weights; recast to the pipeline dtype in case use_f64 is on.
    for net in (fit_sys.encoder, fit_sys.hfn):
        net.to(cfg.dtype_pt)
