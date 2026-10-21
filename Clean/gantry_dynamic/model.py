"""Model construction (interconnect + encoder) and the least-squares encoder start."""
__project_origin__ = "added"

from contextlib import contextmanager

import numpy as np
import torch

from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.utils.torch_nets import live_g_feed_forward_nn
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect_Composed
from model_augmentation.fit_systems.blocks import Reduced_Gantry_State_Block, Linear_Output_Block, Static_ANN_Block
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout
from model_augmentation.systems import gantry_ss

from .config import RunConfig
from .obc_gantry import attach_obc, P_INV_T64

PHY_IX   = np.arange(6)


def build_model(hp, cfg: RunConfig, data, norm):
    """Build interconnect + SSE_Interconnect from hp dict, return fit_sys (untrained)."""
    NX_ANN = hp['NX_ANN']
    nxd = cfg.nx_phys + NX_ANN
    route_ix = np.asarray(cfg.ann_route_ix)

    # THEORY: Hoekstra 2026 Eq. (22e), x_k|k from y and u over k-na .. k-1 (D-250): na = nb
    # past samples, the current one excluded.
    na = nb = hp['na_nb']

    ic = Interconnect(nxd, cfg.nu, cfg.ny)

    # CHANGED (D-191): detune and train in the selected coordinate system.
    _names = Reduced_Gantry_State_Block.PARAM_NAMES
    _nominal = torch.stack([getattr(gantry_ss, n) for n in _names]).to(cfg.dtype_pt)
    _combo = Reduced_Gantry_State_Block.combos_of(_nominal, gantry_ss.Lb)
    if cfg.combo_init_detune is not None:
        _combo = _combo * _combo.new_tensor(cfg.combo_init_detune)
    phy_block = Reduced_Gantry_State_Block(
        params_init=_nominal, combo_init=_combo, Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=hp['up_sample'],
        RMSE_baseline=cfg.param_rmse_baseline).to(cfg.dtype_pt)
    # CHANGED: C_aug / Parameterized_Linear_Output_Block removed: diag_gradient_routing
    # confirms near-zero C_aug init (Frobenius ~1e-2) chokes ANN gradient by same factor.
    out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_phys)

    ann_block = Static_ANN_Block(
        nz=nxd + cfg.nu, nw=len(route_ix),
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=live_g_feed_forward_nn,
        net_kwargs={'nx_phys': cfg.nx_phys},
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nxd))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_phys, ["u"], ["y"])

    # CHANGED (2026-08-28): one composed class, replacing the
    # ParamLoss -> OrthLoss -> MultipleShooting chain.
    fit_sys = SSE_Interconnect_Composed(interconnect=ic, na=na, nb=nb)

    fit_sys.norm.u0   = norm.u_mean.flatten()
    fit_sys.norm.ustd = norm.std_u.flatten()
    fit_sys.norm.y0   = norm.y0
    fit_sys.norm.ystd = norm.ystd

    # D-251/D-271: physical rows set to the least-squares map of a closed-loop baseline
    # simulation by prefit_physical_encoder before training. The class reads n+1 samples for its
    # argument n, so n = na - 1 gives the na-sample window.
    fit_sys.encoder = linear_encoder_init_aug(
        nx=cfg.nx_phys, nu=cfg.nu, ny=cfg.ny, na=na - 1, nb=nb - 1,
        nx_aug=NX_ANN,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        dtype=cfg.dtype_pt,
    ).to(cfg.dtype_pt)

    # CHANGED: pass lr and eps at optimizer creation.
    # CHANGED (D-169, job 80714): the model is BUILT on the CPU even when cfg.device='cuda', and
    # train moves it for the duration of fit() only.
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'device="cuda" was requested but torch.cuda.is_available() is False. Falling back to '
            'the CPU is deliberately NOT done: the run would be slower by more than an order of '
            'magnitude while config.json recorded DEVICE=cuda, and the two would be compared as '
            'if they were the same experiment. Check the SLURM allocation (--gres=gpu:1) or set '
            'device="cpu" explicitly.')
    fit_sys.init_model(sys_data=data.train_data, optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps})
    eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
    if eps_values != {float(cfg.adam_eps)}:
        raise RuntimeError(
            f'Adam eps mismatch after optimizer construction: config={cfg.adam_eps}, '
            f'optimizer={sorted(eps_values)}')
    fit_sys.hfn.to(cfg.dtype_pt)

    if cfg.obc:
        attach_obc(fit_sys, cfg, data, norm)

    return fit_sys


@contextmanager
def on_device(fit_sys, cfg: RunConfig):
    """D-169: the model lives on the CPU and visits cfg.device for the work inside."""
    if cfg.device == 'cuda':
        fit_sys.cuda()
    try:
        yield
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()


def prefit_physical_encoder(fit_sys, cfg: RunConfig, data, norm):
    """D-251, D-271: start the PHYSICAL rows of the encoder at the least-squares map of a closed-loop
    baseline simulation."""
    with on_device(fit_sys, cfg):
        _prefit(fit_sys, cfg, data, norm)


def _prefit(fit_sys, cfg, data, norm):
    enc, sim = fit_sys.encoder, fit_sys.simulator
    nx, na, nb, stride = cfg.nx_phys, fit_sys.na, fit_sys.nb, cfg.stride
    dtype, dev = next(enc.parameters()).dtype, next(enc.parameters()).device
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)
    rv = lambda a: np.asarray(a, np.float64).ravel()
    fit_sys._sync_prediction_state_for_validation()
    step_kw = {}
    with torch.no_grad():
        step_kw['pass_mats'] = fit_sys.hfn.connected_blocks[fit_sys._mats_ix].pass_mats()
        corr = getattr(fit_sys.hfn, 'obc_correction', None)
        if corr is not None:
            step_kw['obc_pass'] = corr.tangent_pair(corr.theta_frozen)

    sdl = data.train_data.sdl
    x_mean, std_x = rv(norm.x_mean), rv(norm.std_x)
    uh_all, yh_all, xt_all = [], [], []
    groups = {}
    for i, sd in enumerate(sdl):
        groups.setdefault(len(sd.u), []).append(i)
    for ixs in groups.values():
        un = np.stack([(sdl[i].u - rv(fit_sys.norm.u0)) / rv(fit_sys.norm.ustd) for i in ixs])
        yn = np.stack([(sdl[i].y - rv(fit_sys.norm.y0)) / rv(fit_sys.norm.ystd) for i in ixs])
        x0 = np.zeros((len(ixs), nx + cfg.nx_ann))
        q0 = np.stack([np.asarray(sdl[i].y[0], np.float64) @ P_INV_T64.T for i in ixs])
        x0[:, :nx] = (np.hstack([q0, np.zeros_like(q0)]) - x_mean) / std_x
        xs, us = [], []

        def recorder(x, u, *args):
            xs.append(x)
            us.append(u)
            return fit_sys.hfn(x, u, *args)

        with torch.no_grad():
            y_hat, _, _ = closed_loop_rollout(
                recorder, fit_sys.hfn.output_only, T(un), T(yn), T(x0), sim.bank,
                torch.tensor([sim.train_ctrl_rows[i] for i in ixs], dtype=torch.long, device=dev),
                **step_kw)
        X = torch.stack(xs, 1).reshape(len(ixs), len(xs), -1)
        U = torch.stack(us, 1).reshape(len(ixs), len(us), -1)
        starts = torch.arange(0, X.shape[1] - max(na, nb), stride, device=dev)
        uh_all.append(U[:, starts[:, None] + torch.arange(nb, device=dev)].flatten(0, 1))
        yh_all.append(y_hat[:, starts[:, None] + torch.arange(na, device=dev)].flatten(0, 1))
        xt_all.append(X[:, starts + max(na, nb), :nx].flatten(0, 1))
    uh, yh, xt = torch.cat(uh_all), torch.cat(yh_all), torch.cat(xt_all)
    print('[encoder start] %d windows from %d records' % (len(xt), len(sdl)), flush=True)
    least_squares_start(enc, nx, uh, yh, xt)


# HEURISTIC (D-271, Dirk): readiness of the start, every physical state reconstructed with R^2 >= 0.999 on
# the simulated windows; the accurate start of the ablation (F1) reached about 1e-7, the failing ones 0.04 to 8.8.
_START_MAX_UNEXPLAINED = 1e-3


def least_squares_start(enc, nx, uh, yh, xt):
    """D-271: set the physical rows of the encoder to the least-squares map x ~ W [yh; uh] + b."""
    final = enc.net[-1]
    wa_before = (enc.Wa_psi_y.detach().clone(), enc.Wa_psi_u.detach().clone())
    cpu = lambda a: a.detach().to('cpu', torch.float64)
    Phi, X = cpu(torch.cat([yh.flatten(1), uh.flatten(1)], 1)), cpu(xt)
    if not (torch.isfinite(Phi).all() and torch.isfinite(X).all()):
        raise RuntimeError('the encoder start windows contain non-finite values')
    mu, s, x_mu = Phi.mean(0), Phi.std(0, unbiased=False), X.mean(0)
    s = torch.where(s > 0, s, torch.ones_like(s))
    rcond = torch.finfo(torch.float64).eps * max(Phi.shape)
    sol = torch.linalg.lstsq((Phi - mu) / s, X - x_mu, rcond=rcond, driver='gelsd')
    W = sol.solution / s[:, None]
    b = x_mu - mu @ W
    pred = Phi @ W + b
    if not (torch.isfinite(W).all() and torch.isfinite(b).all() and torch.isfinite(pred).all()):
        raise RuntimeError('the least-squares encoder start is not finite')
    ny_h = yh[0].numel()
    with torch.no_grad():
        p = next(enc.parameters())
        enc.Wb_psi_y.copy_(W[:ny_h].T.to(p))
        enc.Wb_psi_u.copy_(W[ny_h:].T.to(p))
        final.weight[:nx] = 0.0
        final.bias[:nx] = 0.0
        final.bias[:nx] = b.to(p) - enc(uh[:1] * 0, yh[:1] * 0)[0, :nx]
        got = cpu(enc(uh, yh)[:, :nx])
    unexplained = ((pred - X) ** 2).mean(0) / X.var(0, unbiased=False)
    print('[encoder start] least squares: %d regressors, rank %d, condition %.3e; unexplained per state %s'
          % (Phi.shape[1], int(sol.rank), float(sol.singular_values[0] / sol.singular_values[-1]),
             ' '.join('%.2e' % v for v in unexplained)), flush=True)
    if not torch.allclose(got, pred, rtol=0.0, atol=1e-6 * float(X.std(0, unbiased=False).min())):
        raise RuntimeError('the encoder does not reproduce the least-squares map (max deviation %.3e)'
                           % float((got - pred).abs().max()))
    if not (unexplained <= _START_MAX_UNEXPLAINED).all():
        raise RuntimeError('encoder start not ready: a physical state is reconstructed with R^2 < %.3f (%s)'
                           % (1 - _START_MAX_UNEXPLAINED, ' '.join('%.2e' % v for v in unexplained)))
    if not (torch.equal(wa_before[0], enc.Wa_psi_y) and torch.equal(wa_before[1], enc.Wa_psi_u)
            and not final.weight[nx:].any() and not final.bias[nx:].any()):
        raise RuntimeError('the encoder start moved W^a or an added-state output row')
    return W, b
