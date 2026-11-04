"""Builds the augmented model (baseline, network, wiring, encoder) and gives the encoder its least-squares start from
a baseline simulation."""
__project_origin__ = "added"

from contextlib import contextmanager

import deepSI
import numpy as np
import torch

from model_augmentation.utils.utils import expansion_matrix, selection_matrix
from model_augmentation.utils.torch_nets import AugmentationNetwork
from model_augmentation.fit_systems.interconnect import Interconnect, AugmentedModel
from model_augmentation.fit_systems.blocks import IdentifiableGantryStateBlock, Linear_Output_Block, Static_ANN_Block
from model_augmentation.fit_systems.pre_encoder import AugmentedEncoder
from model_augmentation.fit_systems.closed_loop import closed_loop_rollout, closed_loop_step
from model_augmentation.systems import gantry_ss
from model_augmentation.systems.gantry_ss import P_INV_T

from .config import RunConfig
from .obc_gantry import attach_obc


def build_model(cfg: RunConfig, data, norm):
    """The untrained augmented model, which starts as the baseline (beta = 0: theta_base = theta^0_base;
    f_aug^n = 0), one step per T_s in normalised coordinates.

    Args:
        cfg: run configuration.
        data: the training and validation records (Datasets).
        norm: means and standard deviations of the training records (Norm).
    Returns:
        the model with its Adam optimizer and, on the OBC arm, the projection.
    """
    # model state size: physical states plus n_a added states
    nx = cfg.nx_phys + cfg.n_a

    # na = nb past samples before the window start, the current sample excluded
    na = nb = cfg.na_nb

    # 1. Baseline block (physics, f_base^n) and output map C^n; nu here is the number of inputs, not the PS score.
    interconnect = Interconnect(nx, cfg.nu, cfg.ny)

    # start values theta^0_base of the 10 identifiable parameter combinations, from the nominal parameters of
    # gantry_ss; with the detuned start, each combination times 1 +- detune (cfg.combo_init_detune)
    parameter_names = IdentifiableGantryStateBlock.PARAM_NAMES
    nominal_parameters = torch.stack([getattr(gantry_ss, n) for n in parameter_names]).to(cfg.dtype_pt)
    initial_combinations = IdentifiableGantryStateBlock.combos_of(nominal_parameters, gantry_ss.Lb)
    if cfg.combo_init_detune is not None:
        initial_combinations = initial_combinations * initial_combinations.new_tensor(cfg.combo_init_detune)
    baseline_block = IdentifiableGantryStateBlock(
        params_init=nominal_parameters, combo_init=initial_combinations, std_x=norm.std_x, std_u=norm.std_u,
        x_mean=norm.x_mean, u_mean=norm.u_mean, Ts=cfg.ts_new, up_sample=cfg.up_sample,
        closed_form=cfg.baseline_form == 'closed').to(cfg.dtype_pt)
    output_block = Linear_Output_Block(C=norm.Cd_norm, D=norm.Dd_np)
    # put both blocks in the model; step 3 connects them
    interconnect.add_block(baseline_block)
    interconnect.add_block(output_block)

    # 2. Network block phi_aug: (state, input) -> corrections f_aug^n to the physical states and the next added
    # states g_aug^n; f_aug^n starts at zero, so the model starts as the baseline.
    ann_block = Static_ANN_Block(
        nz=nx + cfg.nu, nw=nx,
        n_nodes_per_layer=cfg.n_nodes_per_layer,
        n_hidden_layers=cfg.n_hidden_layers,
        net=AugmentationNetwork,
        net_kwargs={'nx_phys': cfg.nx_phys},
        activation=torch.nn.Tanh,
    )
    interconnect.add_block(ann_block)

    # 3. Wiring, with x_hat^n = (x_tilde^n, x_bar):
    #   x_tilde^n_{k+1} = f_base^n(x_tilde^n_k, u^n_k) + f_aug^n(x_hat^n_k, u^n_k)   (- f_obc^n on the OBC arm)
    #   x_bar_{k+1} = g_aug^n(x_hat^n_k, u^n_k),   y_hat^n_k = C^n x_tilde^n_k
    # 'additive' adds a block's output into a signal through an expansion matrix; 'concat' feeds the selected rows of
    # a signal into a block
    physical_rows = np.arange(cfg.nx_phys)  # rows of x_tilde^n in x_hat^n
    # network: reads the whole state and the input, adds into every state
    interconnect.connect_block_signals(ann_block, ["x", "u"], [])
    interconnect.connect_signals(ann_block, "xp", "additive", expansion_matrix(np.arange(nx), nx))
    # baseline: reads the physical states and the input, adds into the physical states
    interconnect.connect_signals("x", baseline_block, "concat", selection_matrix(physical_rows, nx))
    interconnect.connect_block_signals(baseline_block, ["u"], [])
    interconnect.connect_signals(baseline_block, "xp", "additive", expansion_matrix(physical_rows, nx))
    # output map: reads the physical states, gives y
    interconnect.connect_signals("x", output_block, "concat", selection_matrix(physical_rows, nx))
    interconnect.connect_block_signals(output_block, ["u"], ["y"])

    # 4. Model, normaliser and encoder.
    fit_sys = AugmentedModel(interconnect=interconnect, na=na, nb=nb)

    # normaliser of the data windows: u^n = (u_act - mu_u) / sigma_u and y^n = (y - mu_y) / sigma_y, the same means and
    # standard deviations as the baseline's normalisation
    fit_sys.norm.u0   = norm.u_mean.flatten()
    fit_sys.norm.ustd = norm.std_u.flatten()
    fit_sys.norm.y0   = norm.y0
    fit_sys.norm.ystd = norm.ystd

    # encoder psi(chi) = [W^b; W^a] chi + psi_tilde(chi), psi_tilde a network; its window holds na + 1 samples of the
    # given na, so na - 1 gives the na samples of the model
    fit_sys.encoder = AugmentedEncoder(
        nx=cfg.nx_phys, nu=cfg.nu, ny=cfg.ny, na=na - 1, nb=nb - 1,
        nx_aug=cfg.n_a,
        n_nodes_per_layer=cfg.n_nodes_per_layer,
        n_hidden_layers=cfg.n_hidden_layers,
        dtype=cfg.dtype_pt,
    ).to(cfg.dtype_pt)

    # 5. Optimizer and, on the OBC arm, the projection.
    # Adam over every parameter (encoder and interconnect), learning rate cfg.lr; the normaliser keeps the values set
    # above
    fit_sys.init_model(sys_data=deepSI.System_data_list(data.train_list),
                       optimizer_kwargs={'lr': cfg.lr, 'eps': cfg.adam_eps})
    # cast the interconnect to the run's dtype: torch builds the network in its default float32
    fit_sys.hfn.to(cfg.dtype_pt)

    # attach f_obc^n before encoder_start. That start simulates the model and fits the
    # encoder to the simulated states; on the OBC arm the trained model includes f_obc^n, so the simulation must too,
    # else the encoder starts fitted to a different model
    if cfg.obc:
        attach_obc(fit_sys, baseline_block, cfg, data, norm)

    return fit_sys


@contextmanager
def on_device(fit_sys, cfg: RunConfig):
    """Run the encoder start on cfg.device: move the model there, and back to the CPU afterwards, also on an error;
    the model is built and kept on the CPU."""
    if cfg.device == 'cuda':
        fit_sys.cuda()
    try:
        yield
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()


def encoder_start(fit_sys, cfg: RunConfig, data, norm):
    """Encoder start: simulate the untrained model (the baseline at theta^0_base, f_aug^n = 0) in closed loop on
    every training record, then fit the encoder's physical rows to the simulated states, so the encoder starts
    consistent with that model.

    Args:
        fit_sys: the augmented model; its encoder's physical rows are fitted in place.
        cfg: run configuration.
        data: the records (Datasets); the training records are simulated.
        norm: means and standard deviations of the training records (Norm).
    """
    with on_device(fit_sys, cfg):
        encoder, simulator = fit_sys.encoder, fit_sys.simulator
        nx, na, nb, stride = cfg.nx_phys, fit_sys.na, fit_sys.nb, cfg.stride
        dtype, device = next(encoder.parameters()).dtype, next(encoder.parameters()).device
        to_torch = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
        to_flat_numpy = lambda a: np.asarray(a, cfg.dtype_np).ravel()
        # OBC arm: compute the basis J and theta_aux (least-squares coefficients of the network output on J) of the
        # untrained model, which f_obc^n reads in the rollout; nothing on the U arm
        fit_sys.sync_obc_coefficient()
        with torch.no_grad():
            # baseline matrices (stiffness K, damping C, M(Y)^-1 parts) and on the OBC arm those of f_obc^n, built
            # once for the rollout
            matrices = fit_sys.objective_matrices()

        # 1. Simulate the untrained model in closed loop, as training rolls it out (with f_obc^n on the OBC arm);
        #    records of the same length are simulated together, as one batch: the steps run one after another, the
        #    records of a batch in parallel on the GPU.
        records = data.train_list
        x_mean, std_x = to_flat_numpy(norm.x_mean), to_flat_numpy(norm.std_x)
        u0, ustd = to_flat_numpy(norm.u_mean), to_flat_numpy(norm.std_u)
        y0, ystd = to_flat_numpy(norm.y0), to_flat_numpy(norm.ystd)
        input_windows, output_windows, state_targets = [], [], []
        groups = {}
        for i, record in enumerate(records):
            groups.setdefault(len(record.u), []).append(i)
        for group in groups.values():
            # u^n = (u_act - mu_u) / sigma_u and y^n = (y - mu_y) / sigma_y of the group's records
            un = np.stack([(records[i].u - u0) / ustd for i in group])
            yn = np.stack([(records[i].y - y0) / ystd for i in group])
            # start state: q_0 = P^-T y_0, qdot_0 = 0, xbar_0 = 0, normalised; qdot_0 = 0 because every record
            # starts with a 0.5 s hold at rest (t_hold in data_generation/tdg_config.m), far longer than the trim
            x0 = np.zeros((len(group), nx + cfg.n_a))
            q0 = np.stack([np.asarray(records[i].y[0], cfg.dtype_np) @ P_INV_T.T for i in group])
            x0[:, :nx] = (np.hstack([q0, np.zeros_like(q0)]) - x_mean) / std_x
            # record each state and force of the simulation: closed_loop_rollout returns only the outputs, so recorder
            # wraps the model and saves its inputs x and u at every step
            xs, us = [], []

            def recorder(x, u, *args):
                """The model step, recording x_hat^n_k and the closed-loop input u_hat^n_k."""
                xs.append(x)
                us.append(u)
                return fit_sys.hfn(x, u, *args)
            # closed_loop_step calls recorder.output_only(x) for y_hat = C x, the output the controller needs at
            # each step; recorder uses the model's own function for it
            recorder.output_only = fit_sys.hfn.output_only

            # simulate the group in closed loop from x0, driven by its recorded data, each record with its own
            # controller
            with torch.no_grad():
                y_hat, _, _ = closed_loop_rollout(
                    closed_loop_step(recorder, simulator.bank), to_torch(un), to_torch(yn), to_torch(x0),
                    simulator.bank,
                    torch.tensor([simulator.train_ctrl_rows[i] for i in group], dtype=torch.long, device=device),
                    **matrices)
            # sort the saved states and forces per record: X[r, k] is the state and U[r, k] the force of record r at
            # time step k
            X = torch.stack(xs, 1).reshape(len(group), len(xs), -1)
            U = torch.stack(us, 1).reshape(len(group), len(us), -1)
            # fit windows, one every stride samples: chi^0_tau holds the simulated outputs y_hat^n,0 and forces
            # u_hat^n,0 of the na and nb samples before tau, the target is the simulated physical state x_tilde^n,0_tau
            starts = torch.arange(0, X.shape[1] - max(na, nb), stride, device=device)
            input_windows.append(U[:, starts[:, None] + torch.arange(nb, device=device)].flatten(0, 1))
            output_windows.append(y_hat[:, starts[:, None] + torch.arange(na, device=device)].flatten(0, 1))
            state_targets.append(X[:, starts + max(na, nb), :nx].flatten(0, 1))
        # 2. Fit the encoder's physical rows to the simulated states: the windows of all groups are joined into one
        #    least-squares problem.
        uh, yh, xt = torch.cat(input_windows), torch.cat(output_windows), torch.cat(state_targets)
        print('[encoder start] %d windows from %d records' % (len(xt), len(records)), flush=True)
        least_squares_start(encoder, nx, uh, yh, xt)


def least_squares_start(encoder, nx, uh, yh, xt):
    """Least-squares start of the encoder: its physical rows reconstruct the untrained model's states from the past
    window; W^a and the added-state rows stay unchanged.

    (W^b0, b^b0) = argmin mean ||W^b chi^0_tau + b - x_tilde^n,0_tau||^2, chi^0 = (y history, u history).

    Args:
        encoder: encoder psi; W^b and its physical output rows are overwritten.
        nx: number of physical states.
        uh: (N, n_b, n_u) past input windows u^n.
        yh: (N, n_a, n_y) past output windows y^n.
        xt: (N, nx) simulated physical states x_tilde^n,0 at the window ends; the encoder is fitted to output these.
    Returns:
        W: (n_chi, nx) = (W^b0)^T, and b: (nx,).
    """
    # 1. Solve the least-squares problem.
    output_layer = encoder.net[-1]
    # the solve uses gelsd, an SVD-based solver that drops directions below rcond; torch runs it on the CPU only
    to_cpu = lambda a: a.detach().to('cpu', xt.dtype)
    # one row per window: Phi holds chi^0_tau (the output values, then the input values), X the simulated state
    Phi, X = to_cpu(torch.cat([yh.flatten(1), uh.flatten(1)], 1)), to_cpu(xt)
    if not (torch.isfinite(Phi).all() and torch.isfinite(X).all()):
        raise RuntimeError('the encoder start windows contain non-finite values: the closed-loop simulation of the '
                           'untrained model diverged')
    # least squares on standardised regressors (Phi - mu) / s; the solution W_s is converted for the windows the
    # encoder receives: W = W_s / s, b = x_mu - mu W
    # mu, s: mean and standard deviation of each column of Phi; x_mu: mean of X
    mu, s, x_mu = Phi.mean(0), Phi.std(0, unbiased=False), X.mean(0)
    # a column that never changes has s = 0; give it scale 1, so it is not divided by zero
    s = torch.where(s > 0, s, torch.ones_like(s))
    # rcond: singular values below rcond times the largest count as zero; eps max(M, N), the numpy.linalg.lstsq default
    rcond = torch.finfo(Phi.dtype).eps * max(Phi.shape)
    # W_s = argmin ||((Phi - mu) / s) W_s - (X - x_mu)||_F^2
    sol = torch.linalg.lstsq((Phi - mu) / s, X - x_mu, rcond=rcond, driver='gelsd')
    W = sol.solution / s[:, None]
    b = x_mu - mu @ W
    pred = Phi @ W + b  # the fitted states
    if not (torch.isfinite(W).all() and torch.isfinite(b).all() and torch.isfinite(pred).all()):
        raise RuntimeError('the least-squares encoder start is not finite')
    # 2. Write W^b into the encoder and set psi_tilde's physical rows to the constant b.
    # each row of Phi holds a window's past outputs, then its past inputs; split W the same way, into the parts for
    # Wb_psi_y and Wb_psi_u
    W_y, W_u = W.split([yh[0].numel(), uh[0].numel()])
    with torch.no_grad():
        encoder.Wb_psi_y.copy_(W_y.T)
        encoder.Wb_psi_u.copy_(W_u.T)
        # the encoder starts as the fit, psi_b = W^b chi + b: psi_tilde's physical rows get output weights 0 and
        # bias b, so they add b for every window; the nonlinear correction psi_tilde is learned during training
        output_layer.weight[:nx] = 0.0
        output_layer.bias[:nx] = b.to(output_layer.bias)
    # 3. Log the fit quality, so a poor encoder start shows in the run log: per state 1 - R_j^2 =
    #    mean((pred_j - X_j)^2) / var(X_j), with the rank and condition of the solve.
    unexplained = ((pred - X) ** 2).mean(0) / X.var(0, unbiased=False)
    print('[encoder start] least squares: %d regressors, rank %d, condition %.3e; unexplained per state %s'
          % (Phi.shape[1], int(sol.rank), float(sol.singular_values[0] / sol.singular_values[-1]),
             ' '.join('%.2e' % v for v in unexplained)), flush=True)
    return W, b
