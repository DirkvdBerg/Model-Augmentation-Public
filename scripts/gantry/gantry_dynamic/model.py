"""Model construction (interconnect + encoder) and the training call.

The augmentation implementation is untouched: `build_model` and `train_model`
are verbatim moves of the pre-refactor functions, with module globals replaced
by explicit (hp, cfg, data, norm) arguments. The star imports mirror the
pre-refactor entry file so no name resolves differently.
"""
__project_origin__ = "added"

import os

import numpy as np
import torch
import deepSI

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

from .config import RunConfig, REPO_ROOT

PHY_IX   = np.arange(6)             # [0,1,2,3,4,5]  (NX_PHYS physical states)
STIFF_IX = np.array([1, 4, 6, 7])   # Theta pos/vel + absorber pos/vel (K>0 only)


def get_encoder_dims(hp, cfg: RunConfig):
    """Return (na, nb, na_right, nb_right) consistent with build_model logic."""
    NX_PHYS = cfg.nx_phys
    if cfg.encoder_init == 'linear_map':
        na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)  # THEORY: nxd*2+1 (Jan's standard)
        nb = na
        na_right = 1   # reconstructability map uses y(k), so window is [k-na, k]
        nb_right = 1
    else:
        na = hp.get('na_nb', 2 * (NX_PHYS + hp['NX_ANN']) + 1)
        nb = na
        na_right = 0
        nb_right = 0
    return na, nb, na_right, nb_right


def build_model(hp, cfg: RunConfig, data, norm):
    """Build interconnect + SSE_Interconnect from hp dict, return fit_sys (untrained)."""
    NX_PHYS = cfg.nx_phys
    nu, ny  = cfg.nu, cfg.ny
    TS_NEW  = cfg.ts_new
    DTYPE_PT = cfg.dtype_pt
    std_x, std_u = norm.std_x, norm.std_u
    x_mean, u_mean = norm.x_mean, norm.u_mean
    Cd_norm, Dd_np = norm.Cd_norm, norm.Dd_np
    y0, ystd = norm.y0, norm.ystd

    NX_ANN = hp['NX_ANN']
    nxd = NX_PHYS + NX_ANN

    # Encoder history: na=nb=nxd*2+1 (Jan's standard, nxd=NX_PHYS+NX_ANN).
    # For linear_map, na_right=1 so reconstructability map can use y(k) to compute x(k).
    na, nb, na_right, nb_right = get_encoder_dims(hp, cfg)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    if cfg.joint_estimation:
        # D-077: all 14 raw physical params trainable via log-reparam.
        # PARAM_INIT_DETUNE, when set, is a 14-vector aligned to PARAM_NAMES.
        _pi = None
        if cfg.param_init_detune is not None:
            from model_augmentation.systems import gantry_ss as _gss
            _names = Parameterized_Gantry_State_Block.PARAM_NAMES
            _nominal = np.array([float(getattr(_gss, n)) for n in _names])
            _pi = _nominal * np.asarray(cfg.param_init_detune, dtype=float)
        phy_block = Parameterized_Gantry_State_Block(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
            up_sample=hp['up_sample'],
            RMSE_baseline=cfg.param_rmse_baseline,
            params_init=_pi,
        ).to(DTYPE_PT)
    else:
        phy_block = Gantry_State_Block(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
            up_sample=hp['up_sample'],
        ).to(DTYPE_PT)
    # CHANGED: C_aug / Parameterized_Linear_Output_Block removed — diag_gradient_routing
    # confirms near-zero C_aug init (Frobenius ~1e-2) chokes ANN gradient by same factor.
    out_phys = Linear_Output_Block(C=Cd_norm, D=Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_phys)

    _act = torch.nn.Identity if cfg.ann_activation == 'linear' else torch.nn.Tanh
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=len(STIFF_IX),  # D-068: stiff routing — 4 outputs at K>0 rows only
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
    )
    ic.add_block(ann_block)

    # D-068: stiff routing — corrections placed only at K>0 rows (Theta + absorber).
    # X and Y axis rows (0,3,2,5) excluded: K=0 integrators accumulate without restoring force.
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(STIFF_IX, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    # Physical: x_phys + u -> y (additive)
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])

    # D-076: ParamLoss subclass used unconditionally — exact no-op when no
    # block exposes param_loss (i.e. identical to SSE_Interconnect for
    # JOINT_ESTIMATION=False).
    fit_sys = SSE_Interconnect_ParamLoss(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )

    # Manual normalisation: Gantry_State_Block is nonlinear, auto_fit_norm=True would break this.
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # --- Encoder injection (BEFORE init_model) ---
    if cfg.encoder_init == 'linear_map':
        # THEORY: Hoekstra 2026 Eq. 16-17 — initialize encoder from reconstructability map
        Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

        # Build System_data_with_x for normalization (needs state trajectories)
        baseline_npz_path = os.path.join(
            REPO_ROOT, 'data', 'gantry',
            'baseline_simulations', f'{cfg.mode}_LPV', 'baseline_states.npz')
        if os.path.exists(baseline_npz_path):
            bl = np.load(baseline_npz_path, allow_pickle=True)
            x_phys_all = np.concatenate(bl['x_train_phys'])
        else:
            # Fallback: use the finite-diff states from the data loading section
            x_phys_all = norm.x_all
            print("WARNING: baseline_states.npz not found, using finite-diff states for normalization")

        sys_data_with_x = deepSI.System_data(u=norm.u_all, y=norm.y_all)
        sys_data_with_x.x = x_phys_all

        Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
            Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

        # CHANGED: D-055 -- single encoder replaces linear_encoder_init + LinearInitEncoderWrapper.
        # na/nb passed without right extension: W^b is built for (na+1) timesteps, and
        # the SSE_Interconnect window with na_right=1 gives exactly (na+1) timesteps.
        # Convention fix (D-017) applied natively via u_mean/std_u/y0/ystd/x_mean/std_x.
        fit_sys.encoder = linear_encoder_init_aug(
            A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
            nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
            nx_aug=NX_ANN,
            n_nodes_per_layer=hp['n_nodes_per_layer'],
            n_hidden_layers=hp['n_hidden_layers'],
            flag_linear_only=False,
            u_mean=u_mean, std_u=std_u,
            y0=y0, ystd=ystd,
            x_mean=x_mean, std_x=std_x,
        ).to(DTYPE_PT)

    fit_sys.init_model(sys_data=data.train_data, auto_fit_norm=False)
    if cfg.encoder_init == 'linear_map':
        fit_sys.hfn.to(DTYPE_PT)
    else:
        for net in (fit_sys.encoder, fit_sys.hfn):
            net.to(DTYPE_PT)

    return fit_sys


def train_model(fit_sys, hp, cfg: RunConfig, data, epochs=None, nf=None, validation_measure=None):
    """Train fit_sys for given epochs. nf and validation_measure override hp defaults."""
    fit_sys.fit(
        train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
        batch_size=hp['batch_size'], epochs=epochs or hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': nf if nf is not None else hp['nf'], 'stride': cfg.stride},
        optimizer_kwargs={'lr': hp['lr']},
        validation_measure=validation_measure if validation_measure is not None else 'sim-RMS',
    )
    return fit_sys.bestfit
