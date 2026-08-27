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
from model_augmentation.fit_systems.orth_projection import SSE_Interconnect_OrthLoss
from model_augmentation.fit_systems.multiple_shooting import SSE_Interconnect_MultipleShooting
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

from .config import RunConfig

PHY_IX   = np.arange(6)             # [0,1,2,3,4,5]  (NX_PHYS physical states)
# ANN routing rows are configured via cfg.ann_route_ix (default (1,4,6,7) = Theta
# pos/vel + absorber pos/vel, K>0 only; D-068). State layout (logical coords):
#   [X, Theta, Y, dX, dTheta, dY, delta_a, vdelta_a] = idx 0..7.


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
    route_ix = np.asarray(cfg.ann_route_ix)   # ANN correction rows (D-068 default (1,4,6,7))

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
        nz=nxd + nu, nw=len(route_ix),  # D-068: ANN outputs, one per routed row (cfg.ann_route_ix)
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
    )
    ic.add_block(ann_block)

    # ReZero-style gate: zero ANN OUTPUT at init WITHOUT a zero output PROJECTION, so the branch
    # F(z) is live and the gate alpha receives gradient at step 1 (arXiv:2003.04887; the degenerate
    # zero-projection-plus-zero-gate saddle is arXiv:2607.16568). Addresses the W^a dead zone
    # (D-130, gate G1).
    if os.environ.get('ANN_REZERO_GATE'):
        from .rezero_gate import apply_rezero_gate
        _alpha = apply_rezero_gate(ann_block.net)
        print(f"[rezero] final layer re-initialised + zero scalar gate (alpha={float(_alpha):.1f}); "
              f"ANN output is still exactly zero at init")

    # REMOVED (2026-08-13): the D-118 Lipschitz-cap hook (env ANN_LIPSCHITZ -> apply_lipschitz_cap)
    # used to sit here. It was a one-off experiment hook in the production model builder, and the
    # experiment is closed: the only run that ever set it (boae5mdee, 2026-07-18/19,
    # problem-log line 606) found L=1.0 NON-BINDING (the trained ANN's natural Lipschitz is ~1e-2),
    # so the sweep was inconclusive and the signal pointed to the passivity route, not magnitude
    # capping. `lipschitz.py` is kept: to re-run the sweep, call apply_lipschitz_cap on the ANN
    # block from the sweep script AFTER build_model returns, e.g.
    #   ann = next(m for m in fit_sys.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    #   apply_lipschitz_cap(ann.net, L)
    # An experiment hook belongs in the experiment script, not in every run's build path.

    # D-068: stiff routing — corrections placed only at K>0 rows (Theta + absorber).
    # X and Y axis rows (0,3,2,5) excluded: K=0 integrators accumulate without restoring force.
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    # Physical: x_phys + u -> y (additive)
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])

    # D-076: ParamLoss subclass used unconditionally — exact no-op when no
    # block exposes param_loss (i.e. identical to SSE_Interconnect for
    # JOINT_ESTIMATION=False).
    # CHANGED (orth-projection D7.1): OrthLoss subclass of ParamLoss, same
    # unconditional-no-op pattern; identical to ParamLoss while orth_penalty
    # stays None (attached below only when cfg.orth_beta > 0).
    # CHANGED (multiple shooting D-127): MultipleShooting subclass of OrthLoss, same
    # unconditional-no-op pattern; identical to OrthLoss while n_seg==1 and
    # defect_weight==0 (the defaults), so every existing run is bit-identical.
    #
    # STATUS (2026-08-13): multiple shooting is NOT USED by any production training run.
    # cfg defaults are n_seg=1, defect_weight=0, defect_acc_weight=0, i.e. the no-op path
    # (verified bit-identical by coulomb-offset/verify_ms_class.py, contract E4). It is kept
    # only because the diagnostic scripts below construct cfg with n_seg>1 and read the
    # fit_sys attributes set here:
    #   coulomb-offset/{verify_ms_class,verify_ms_gradient,diag_mean_defect}.py
    #   pysynth-data/{check_defect_sees_failure(MS6),check_coherent_defect,
    #                 check_defect_gradient(MS8),check_grad_vs_freerun,measure_defect_split}.py
    #   drift-isolation/t3_true_y_scheduling/build_t3.py  (guards on n_seg==1)
    # CANDIDATE FOR REMOVAL: if those diagnostics are retired, drop the five fit_sys.defect*/
    # n_seg assignments and the print below, revert this base class to
    # SSE_Interconnect_OrthLoss, and remove n_seg/defect_* from RunConfig (config.py) plus the
    # nf/nf_seg split. Do NOT remove it piecemeal: the base class swap is unconditional, so
    # reverting it means re-checking the orth path against the ParamLoss no-op contract.
    fit_sys = SSE_Interconnect_MultipleShooting(
        interconnect=ic, na=na, nb=nb,
        na_right=na_right, nb_right=nb_right,
        e_net_kwargs={
            "n_nodes_per_layer": hp['n_nodes_per_layer'],
            "n_hidden_layers": hp['n_hidden_layers'],
        },
    )
    fit_sys.n_seg         = cfg.n_seg
    fit_sys.defect_weight = cfg.defect_weight
    fit_sys.defect_acc_weight = cfg.defect_acc_weight
    fit_sys.defect_norm   = cfg.defect_norm
    fit_sys.defect_scale  = (None if cfg.defect_scale is None else
                             torch.as_tensor(cfg.defect_scale, dtype=DTYPE_PT))
    if cfg.n_seg > 1 and (cfg.defect_weight != 0.0 or cfg.defect_acc_weight != 0.0):
        print(f'Multiple shooting ON (D-127): n_seg={cfg.n_seg} x nf_seg={cfg.nf_seg} '
              f'= nf {cfg.nf} ({cfg.n_seg * cfg.nf_seconds:.2f} s objective, '
              f'{cfg.nf_seconds:.2f} s gradient path) | '
              f'defect_weight={cfg.defect_weight:g} norm={cfg.defect_norm}')

    # Manual normalisation: Gantry_State_Block is nonlinear, auto_fit_norm=True would break this.
    fit_sys.norm.u0   = u_mean.flatten()
    fit_sys.norm.ustd = std_u.flatten()
    fit_sys.norm.y0   = y0
    fit_sys.norm.ystd = ystd

    # --- Encoder injection (BEFORE init_model) ---
    if cfg.encoder_init == 'linear_map':
        # THEORY: Hoekstra 2026 Eq. 16-17 — initialize encoder from reconstructability map
        Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

        # D-119: the state scaling baked into the encoder matrices must come from
        # the same array that defines x_mean/std_x (norm.x_all), because the D-055
        # x_off subtraction (pre_encoder.py) and the Gantry_State_Block denorm both
        # use norm.std_x; any other std source (the old baseline_states.npz) puts
        # the encoder in a different state frame than the rollout.
        sys_data_with_x = deepSI.System_data(u=norm.u_all, y=norm.y_all)
        sys_data_with_x.x = norm.x_all

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

    # CHANGED: pass lr and eps at optimizer creation. init_model builds the optimizer here;
    # fit()'s optimizer_kwargs are ignored once init_model_done=True, so without this
    # every run trained at Adam's defaults instead of the declared values (D-101/D-148).
    fit_sys.init_model(sys_data=data.train_data, auto_fit_norm=False,
                       optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps})
    eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
    if eps_values != {float(cfg.adam_eps)}:
        raise RuntimeError(
            f'Adam eps mismatch after optimizer construction: config={cfg.adam_eps}, '
            f'optimizer={sorted(eps_values)}')
    if cfg.encoder_init == 'linear_map':
        fit_sys.hfn.to(DTYPE_PT)
    else:
        for net in (fit_sys.encoder, fit_sys.hfn):
            net.to(DTYPE_PT)

    # Orthogonal-projection penalty (D7.1/D7.4): attached when enabled OR in
    # observe mode (orth_observe: beta may be 0; the loss path skips at beta==0,
    # so a beta=0 control keeps the proven no-op loss while the [joint-probe]
    # orth-frac meter can observe). fit_sys.orth_penalty stays None otherwise.
    if cfg.orth_beta > 0 or cfg.orth_observe:
        from .orth_penalty import build_orth_penalty
        fit_sys.orth_penalty = build_orth_penalty(cfg, data, norm)

    return fit_sys


def train_model(fit_sys, hp, cfg: RunConfig, data, epochs=None, nf=None, validation_measure=None):
    """Train fit_sys for given epochs. nf and validation_measure override hp defaults."""
    # A resumed optimizer carries its own parameter-group settings. Refuse a silent
    # fallback to a checkpoint's/default epsilon when the run declares another value.
    eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
    if eps_values != {float(cfg.adam_eps)}:
        raise RuntimeError(
            f'Adam eps mismatch before training: config={cfg.adam_eps}, '
            f'optimizer={sorted(eps_values)}. Resume with the checkpoint epsilon or '
            'start a fresh optimizer.')
    fit_sys.fit(
        train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
        batch_size=hp['batch_size'], epochs=epochs or hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': nf if nf is not None else hp['nf'], 'stride': cfg.stride},
        optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps},
        validation_measure=validation_measure if validation_measure is not None else 'sim-RMS',
    )
    return fit_sys.bestfit
