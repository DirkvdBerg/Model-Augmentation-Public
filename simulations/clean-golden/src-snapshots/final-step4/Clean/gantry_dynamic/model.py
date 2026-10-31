"""Model construction (interconnect and encoder) and the least-squares start of the encoder.

Conventions: superscript n = normalised; chi = (y history, u history); regressions have samples as rows, so the fit
is Phi W ~ X with W = (W^b)^T.
"""
__project_origin__ = "added"

from contextlib import contextmanager

import numpy as np
import torch

from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.utils.torch_nets import AugmentationNetwork
from model_augmentation.fit_systems.interconnect import Interconnect, AugmentedModel
from model_augmentation.fit_systems.blocks import IdentifiableGantryStateBlock, Linear_Output_Block, Static_ANN_Block
from model_augmentation.fit_systems.pre_encoder import AugmentedEncoder
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout, closed_loop_step
from model_augmentation.systems import gantry_ss

from .config import RunConfig
from .obc_gantry import attach_obc, P_INV_T64

# rows of the physical state x_tilde^n (normalised positions and velocities) in the model state x_hat^n
PHYSICAL_STATE_IX = np.arange(6)


def build_model(hp, cfg: RunConfig, data, norm):
    """The untrained augmented model.

    x_tilde^n_{k+1} = f_base^n + f_aug^n, xbar_{k+1} = g_aug^n, y_hat^n = C^n x_tilde^n, with the encoder psi and,
    on the OBC arm, the projection f_obc^n; beta = 0 and f_aug^n = 0, so it starts as the baseline.

    Args:
        hp: hyperparameters; cfg: run configuration; data: records; norm: normalisation.
    Returns:
        The AugmentedModel with its Adam optimizer.
    """
    n_a = hp['NX_ANN']
    nx = cfg.nx_phys + n_a
    route_ix = np.asarray(cfg.ann_route_ix)

    # THEORY: na = nb history samples precede the window start; the current sample is excluded
    na = nb = hp['na_nb']

    # 1. Baseline block and output map.
    ic = Interconnect(nx, cfg.nu, cfg.ny)

    # Start: theta^0_base = theta_base(theta^0) on the correct arm, times (1 +- 0.1) per combination on the detuned
    # arm; beta = 0 there.
    raw_names = IdentifiableGantryStateBlock.PARAM_NAMES
    raw_nominal = torch.stack([getattr(gantry_ss, n) for n in raw_names]).to(cfg.dtype_pt)
    initial_combinations = IdentifiableGantryStateBlock.combos_of(raw_nominal, gantry_ss.Lb)
    if cfg.combo_init_detune is not None:
        initial_combinations = initial_combinations * initial_combinations.new_tensor(cfg.combo_init_detune)
    phy_block = IdentifiableGantryStateBlock(
        params_init=raw_nominal, combo_init=initial_combinations, Y_op=None, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=hp['up_sample'],
        RMSE_baseline=cfg.param_rmse_baseline, closed_form=cfg.baseline_form == 'closed').to(cfg.dtype_pt)
    out_phys = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)
    ic.add_block(phy_block)
    ic.add_block(out_phys)

    # 2. Network block phi_aug.
    ann_block = Static_ANN_Block(
        nz=nx + cfg.nu, nw=len(route_ix),
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=AugmentationNetwork,
        net_kwargs={'nx_phys': cfg.nx_phys},
        activation=torch.nn.Tanh,
    )
    ic.add_block(ann_block)

    # 3. Wiring.
    # the augmentation structure, row by row, with x_hat^n = (x_tilde^n, x_bar) and phi_aug = (f_aug^n, g_aug^n):
    #   x_tilde^n_{k+1} = f_base^n(x_tilde^n_k, u^n_k) + f_aug^n(x_hat^n_k, u^n_k)   (- f_obc^n on the OBC arm)
    #   x_bar_{k+1} = g_aug^n(x_hat^n_k, u^n_k),   y_hat^n_k = C^n x_tilde^n_k   (D = 0)
    ic.connect_block_signals(ann_block, ["x", "u"], [])
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(route_ix, nx))
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHYSICAL_STATE_IX, nx))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHYSICAL_STATE_IX, nx))
    ic.connect_signals("x", out_phys, "concat", selection_matrix(PHYSICAL_STATE_IX, nx))
    ic.connect_block_signals(out_phys, ["u"], ["y"])

    # 4. Model, normaliser and encoder.
    fit_sys = AugmentedModel(interconnect=ic, na=na, nb=nb)

    fit_sys.norm.u0   = norm.u_mean.flatten()
    fit_sys.norm.ustd = norm.std_u.flatten()
    fit_sys.norm.y0   = norm.y0
    fit_sys.norm.ystd = norm.ystd

    # psi(chi) = [W^b; W^a] chi + psi_tilde(chi); prefit_physical_encoder fits its physical rows before training.
    # The class reads n + 1 samples for its argument n, so na - 1 gives the na-sample window.
    fit_sys.encoder = AugmentedEncoder(
        nx=cfg.nx_phys, nu=cfg.nu, ny=cfg.ny, na=na - 1, nb=nb - 1,
        nx_aug=n_a,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        dtype=cfg.dtype_pt,
    ).to(cfg.dtype_pt)

    # 5. Optimizer and, on the OBC arm, the projection.
    # no fallback to the CPU: config.json records the device the run asked for
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'device="cuda" was requested but CUDA is not available; check the GPU allocation '
            '(--gres=gpu:1) or run with --cpu.')
    fit_sys.init_model(sys_data=data.train_data, optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps})
    eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
    if eps_values != {float(cfg.adam_eps)}:
        raise RuntimeError(
            f'Adam eps after optimizer construction is {sorted(eps_values)}, the config asks '
            f'{cfg.adam_eps}; pass the configured eps to init_model through optimizer_kwargs.')
    fit_sys.hfn.to(cfg.dtype_pt)

    if cfg.obc:
        attach_obc(fit_sys, cfg, data, norm)

    return fit_sys


@contextmanager
def on_device(fit_sys, cfg: RunConfig):
    """Move the model, built and kept on the CPU, to cfg.device for the work inside the block."""
    if cfg.device == 'cuda':
        fit_sys.cuda()
    try:
        yield
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()


def prefit_physical_encoder(fit_sys, cfg: RunConfig, data, norm):
    """Encoder start from a baseline simulation.

    x_tilde^n,0_{k+1} = f_base^n(theta^0_base, x_tilde^n,0_k, u_hat^n,0_k) simulated in closed loop on every training
    record, then (W^b, b) fitted to the simulated states, so the encoder starts consistent with the start model.

    Args:
        fit_sys: the augmented model; its encoder's physical rows are fitted in place.
        cfg: run configuration.
        data: the records.
        norm: normalisation.
    Returns:
        None.
    """
    with on_device(fit_sys, cfg):
        enc, sim = fit_sys.encoder, fit_sys.simulator
        nx, na, nb, stride = cfg.nx_phys, fit_sys.na, fit_sys.nb, cfg.stride
        dtype, dev = next(enc.parameters()).dtype, next(enc.parameters()).device
        as_tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)
        as_vector = lambda a: np.asarray(a, np.float64).ravel()
        fit_sys._sync_obc_coefficient()
        with torch.no_grad():
            step_kw = fit_sys.objective_matrices()

        # 1. Simulate the start model in closed loop, record by record (equal lengths batched).
        sdl = data.train_data.sdl
        x_mean, std_x = as_vector(norm.x_mean), as_vector(norm.std_x)
        input_windows, output_windows, state_targets = [], [], []
        groups = {}
        for i, sd in enumerate(sdl):
            groups.setdefault(len(sd.u), []).append(i)
        for ixs in groups.values():
            un = np.stack([(sdl[i].u - as_vector(fit_sys.norm.u0)) / as_vector(fit_sys.norm.ustd) for i in ixs])
            yn = np.stack([(sdl[i].y - as_vector(fit_sys.norm.y0)) / as_vector(fit_sys.norm.ystd) for i in ixs])
            # Start state: q_0 = P^-T y_0, qdot_0 = 0, normalised; xbar_0 = 0 (x^c_0 = 0 in the rollout). The whole
            # start model is simulated: f_aug^n = 0 and f_obc^n = 0 on the physical rows, so x_tilde^n is f_base^n's.
            x0 = np.zeros((len(ixs), nx + cfg.nx_ann))
            q0 = np.stack([np.asarray(sdl[i].y[0], np.float64) @ P_INV_T64.T for i in ixs])
            x0[:, :nx] = (np.hstack([q0, np.zeros_like(q0)]) - x_mean) / std_x
            xs, us = [], []

            def recorder(x, u, *args):
                """The model step, recording x_hat^n_k and the closed-loop input u_hat^n_k = u^n_k + u_fb^n_k."""
                xs.append(x)
                us.append(u)
                return fit_sys.hfn(x, u, *args)
            recorder.output_only = fit_sys.hfn.output_only

            with torch.no_grad():
                y_hat, _, _ = closed_loop_rollout(
                    closed_loop_step(recorder, sim.bank), as_tensor(un), as_tensor(yn), as_tensor(x0), sim.bank,
                    torch.tensor([sim.train_ctrl_rows[i] for i in ixs], dtype=torch.long, device=dev),
                    **step_kw)
            X = torch.stack(xs, 1).reshape(len(ixs), len(xs), -1)
            U = torch.stack(us, 1).reshape(len(ixs), len(us), -1)
            # Fit samples tau = s + k0 at the training stride: chi^0_tau from y_hat^n,0 and u_hat^n,0 at
            # s .. s + k0 - 1, target x_tilde^n,0_tau.
            starts = torch.arange(0, X.shape[1] - max(na, nb), stride, device=dev)
            input_windows.append(U[:, starts[:, None] + torch.arange(nb, device=dev)].flatten(0, 1))
            output_windows.append(y_hat[:, starts[:, None] + torch.arange(na, device=dev)].flatten(0, 1))
            state_targets.append(X[:, starts + max(na, nb), :nx].flatten(0, 1))
        # 2. Fit the encoder to the simulated states.
        uh, yh, xt = torch.cat(input_windows), torch.cat(output_windows), torch.cat(state_targets)
        print('[encoder start] %d windows from %d records' % (len(xt), len(sdl)), flush=True)
        least_squares_start(enc, nx, uh, yh, xt)


# HEURISTIC: require R^2 >= 0.999 for every physical state on the simulated encoder-fit windows
_MAX_ENCODER_UNEXPLAINED_FRACTION = 1e-3


def least_squares_start(enc, nx, uh, yh, xt):
    """Least-squares start of the encoder.

    (W^b0, b^b0) = argmin mean ||W^b chi_tau + b - x_tilde^n_tau||^2.
    Training then starts from an encoder that already reconstructs the start
    model's physical states from the past window. Only the physical rows are
    set; W^a and the added-state rows stay unchanged.

    Args:
        enc: encoder psi (W^b, W^a and psi_tilde); W^b and the physical output rows of psi_tilde are overwritten.
        nx: number of physical states.
        uh: (N, n_b, n_u) past input windows u^n.
        yh: (N, n_a, n_y) past output windows y^n.
        xt: (N, nx) target states x_tilde^n at the window end.
    Returns:
        W: (n_chi, nx) = (W^b0)^T, and b: (nx,).
    """
    # 1. Solve.
    final = enc.net[-1]
    wa_before = (enc.Wa_psi_y.detach().clone(), enc.Wa_psi_u.detach().clone())
    cpu = lambda a: a.detach().to('cpu', torch.float64)
    Phi, X = cpu(torch.cat([yh.flatten(1), uh.flatten(1)], 1)), cpu(xt)
    if not (torch.isfinite(Phi).all() and torch.isfinite(X).all()):
        raise RuntimeError('the encoder start windows contain non-finite values; check the closed-loop '
                           'simulation of the start model')
    # Least squares on centred, standardised regressors: W = W_s / s, b = mean(X) - mean(chi) W.
    mu, s, x_mu = Phi.mean(0), Phi.std(0, unbiased=False), X.mean(0)
    s = torch.where(s > 0, s, torch.ones_like(s))
    # THEORY: rcond = eps max(M, N), the numpy.linalg.lstsq default
    rcond = torch.finfo(torch.float64).eps * max(Phi.shape)
    # the SVD driver, for the rank-truncated least-squares fit
    # W_s = argmin ||((chi - mu) / s) W_s - (X - mean X)||_F^2.
    sol = torch.linalg.lstsq((Phi - mu) / s, X - x_mu, rcond=rcond, driver='gelsd')
    W = sol.solution / s[:, None]
    b = x_mu - mu @ W
    pred = Phi @ W + b
    if not (torch.isfinite(W).all() and torch.isfinite(b).all() and torch.isfinite(pred).all()):
        raise RuntimeError('the least-squares encoder start is not finite; check the encoder-fit windows')
    # 2. Write into the encoder.
    ny_h = yh[0].numel()
    with torch.no_grad():
        p = next(enc.parameters())
        enc.Wb_psi_y.copy_(W[:ny_h].T.to(p))
        enc.Wb_psi_u.copy_(W[ny_h:].T.to(p))
        # Physical rows of psi_tilde's output layer: W~^xtilde_L = 0, b~^xtilde_L = b - psi_b(0) (bias zeroed
        # first), so psi_b(chi) = W^b chi + b.
        final.weight[:nx] = 0.0
        final.bias[:nx] = 0.0
        final.bias[:nx] = b.to(p) - enc(uh[:1] * 0, yh[:1] * 0)[0, :nx]
        got = cpu(enc(uh, yh)[:, :nx])
    # 3. Check: 1 - R_j^2 = mean((pred_j - X_j)^2) / var(X_j) per state; a large value means the linear window
    # map does not reconstruct that state.
    unexplained = ((pred - X) ** 2).mean(0) / X.var(0, unbiased=False)
    print('[encoder start] least squares: %d regressors, rank %d, condition %.3e; unexplained per state %s'
          % (Phi.shape[1], int(sol.rank), float(sol.singular_values[0] / sol.singular_values[-1]),
             ' '.join('%.2e' % v for v in unexplained)), flush=True)
    if not torch.allclose(got, pred, rtol=0.0, atol=1e-6 * float(X.std(0, unbiased=False).min())):
        raise RuntimeError('the encoder does not reproduce the least-squares map (max deviation %.3e); check '
                           'that psi_tilde has the expected output layer' % float((got - pred).abs().max()))
    if not (unexplained <= _MAX_ENCODER_UNEXPLAINED_FRACTION).all():
        raise RuntimeError('encoder start not ready: a physical state is reconstructed with R^2 < %.3f (%s); '
                           'check the encoder window and the start model'
                           % (1 - _MAX_ENCODER_UNEXPLAINED_FRACTION, ' '.join('%.2e' % v for v in unexplained)))
    if not (torch.equal(wa_before[0], enc.Wa_psi_y) and torch.equal(wa_before[1], enc.Wa_psi_u)
            and not final.weight[nx:].any() and not final.bias[nx:].any()):
        raise RuntimeError('the encoder start moved W^a or an added-state output row; it may set the physical '
                           'rows only')
    return W, b
