"""Model construction (interconnect + encoder) and the training call."""
__project_origin__ = "added"

import os

import numpy as np
import torch
import deepSI

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import paper_resnet, zero_init_feed_forward_nn, live_g_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.fit_systems.interconnect import SSE_Interconnect_Composed
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.utils.utils import normalize_linear_ss_matrices

from .config import RunConfig

# torch.compile backend for the training rollout. THE single conditional this codebase allows
# between two implementations (docs/pytorch-optimization-guidelines.md), which is why it is a
# module constant and not a config field: `inductor` is the only backend that FUSES kernels and
# therefore the only one that reduces the ~254 ops per timestep. Measured alternatives, all on
# blade1: `aot_eager` 0.52x (traces without codegen, pure overhead) and `cudagraphs` 0.02x (per
# step capture is the wrong granularity for a callable invoked 12000 times in a loop).
# `inductor` needs a C++ compiler and CUDA capability >= 7.0, so it cannot run on the Windows
# development PC (no MSVC) or on its Quadro P2000 (sm_61). WHICH MODE is a run parameter and
# lives on RunConfig.compile_mode.
COMPILE_BACKEND = 'inductor'

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
    elif cfg.encoder_init == 'data_lsq':
        # THEORY: Hoekstra 2026 Eq. (22e), x_k|k from y and u over k-na .. k-1 (D-250): na = nb
        # past samples, the current one excluded.
        na = nb = hp['na_nb']
        na_right = nb_right = 0
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
        # CHANGED (D-191): detune and train in the selected coordinate system. Coordinate
        # conversion happens once here and never inside the RK4 loop.
        from model_augmentation.systems import gantry_ss as _gss
        _names = Parameterized_Gantry_State_Block.PARAM_NAMES
        _nominal = torch.stack([getattr(_gss, n) for n in _names]).to(DTYPE_PT)
        _common = dict(
            Y_op=None, std_x=std_x, std_u=std_u,
            x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
            up_sample=hp['up_sample'], RMSE_baseline=cfg.param_rmse_baseline,
            # CHANGED (D-193): the prior is now a DECLARED choice rather than the block's
            # inherited default. `param_prior=True` passes the value the block already used, so
            # this line is an exact no-op for every run before it existed.
            flag_loss_reg=cfg.param_prior,
        )
        if cfg.physics_parameterization == 'reduced':
            _combo = Reduced_Gantry_State_Block.combos_of(_nominal, _gss.Lb)
            if cfg.combo_init_detune is not None:
                _combo = _combo * _combo.new_tensor(cfg.combo_init_detune)
            # D-229: the block, not this pipeline, owns the bounded m_diff coordinate. Constructing
            # it after detuning makes eta0 and the local scale s specific to the declared start.
            phy_block = Reduced_Gantry_State_Block(
                params_init=_nominal, combo_init=_combo, **_common).to(DTYPE_PT)
        else:
            _raw_init = _nominal
            if cfg.param_init_detune is not None:
                _raw_init = _raw_init * _raw_init.new_tensor(cfg.param_init_detune)
            phy_block = Parameterized_Gantry_State_Block(
                params_init=_raw_init, **_common).to(DTYPE_PT)
        # D-199: the hoisted M(Y) structure is threaded by BOTH arms, so its optional static-address
        # buffers are attached here rather than in the OBC attach, which only the projected arm
        # reaches. Off by default; see RunConfig.static_pass_buffers.
        if cfg.static_pass_buffers:
            from model_augmentation.fit_systems.obc import StaticPassBuffers
            phy_block.static_pass = StaticPassBuffers('pass_mats')
        if hasattr(phy_block, 'admissibility'):
            # CHANGED (D-191): one construction-time check, outside the simulation hot path.
            with torch.no_grad():
                _admissibility = phy_block.admissibility()
            if not _admissibility['admissible']:
                raise ValueError(
                    'Reduced gantry initialization is not admissible over the scheduling range: '
                    f'{_admissibility}')
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
    # D-240: g_aug network. Live variants receive which output rows route to added states.
    if cfg.g_init == 'live':
        _ann_net = paper_resnet if cfg.ann_arch == 'resnet' else live_g_feed_forward_nn
        _ann_net_kwargs = {'output_state_indices': tuple(int(i) for i in route_ix), 'nx_phys': NX_PHYS}
    else:
        _ann_net, _ann_net_kwargs = zero_init_feed_forward_nn, None
    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=len(route_ix),  # D-068: ANN outputs, one per routed row (cfg.ann_route_ix)
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=_ann_net,
        net_kwargs=_ann_net_kwargs,
        activation=_act,
    )
    ic.add_block(ann_block)

    # ReZero-style gate: zero ANN OUTPUT at init WITHOUT a zero output PROJECTION, so the branch
    # F(z) is live and the gate alpha receives gradient at step 1 (arXiv:2003.04887; the degenerate
    # zero-projection-plus-zero-gate saddle is arXiv:2607.16568). Addresses the W^a dead zone
    # (D-130, gate G1).
    if os.environ.get('ANN_REZERO_GATE'):
        if cfg.g_init == 'live':
            raise ValueError('ANN_REZERO_GATE is a different initialization and cannot be combined '
                             'with live-g initialization')
        from common.rezero_gate import apply_rezero_gate
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

    # CHANGED (2026-08-28): one composed class, replacing the
    # ParamLoss -> OrthLoss -> MultipleShooting chain. Rationale in the class docstring
    # (model_augmentation/fit_systems/interconnect.py). Those three are untouched and still
    # importable for the ~20 diagnostics that build them directly; Jan's benchmarks use plain
    # SSE_Interconnect and are unaffected. Multiple shooting left RunConfig with the chain
    # (D-127 retired), so there are no n_seg / defect_* attributes to set here any more.
    fit_sys = SSE_Interconnect_Composed(
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
            # CHANGED: build the reconstructability map AT the pipeline dtype. The trailing
            # .to(DTYPE_PT) casts the module, but casting float32 Parameters up to float64
            # does not restore digits already discarded, so with use_f64=True the encoder init
            # would carry float32 accuracy inside a float64 run.
            dtype=DTYPE_PT,
            wa_encoder_init=cfg.wa_encoder_init,
        ).to(DTYPE_PT)
    elif cfg.encoder_init == 'data_lsq':
        # D-251/D-271: physical rows set to the least-squares map of a closed-loop baseline
        # simulation by prefit_physical_encoder before training. The class reads n+1 samples for its argument n,
        # so n = na - 1 gives the na-sample window. No mean arguments: no D-055 offsets (FIX-7);
        # A..D are unused by wb_init='random'.
        fit_sys.encoder = linear_encoder_init_aug(
            A=None, B=None, C=None, D=None,
            nx=NX_PHYS, nu=nu, ny=ny, na=na - 1, nb=nb - 1,
            nx_aug=NX_ANN,
            n_nodes_per_layer=hp['n_nodes_per_layer'],
            n_hidden_layers=hp['n_hidden_layers'],
            flag_linear_only=False,
            dtype=DTYPE_PT,
            wa_encoder_init=cfg.wa_encoder_init,
            wb_init='random',
        ).to(DTYPE_PT)

    # CHANGED: pass lr and eps at optimizer creation. init_model builds the optimizer here;
    # fit()'s optimizer_kwargs are ignored once init_model_done=True, so without this
    # every run trained at Adam's defaults instead of the declared values (D-101/D-148).
    # CHANGED (D-169, job 80714): the model is BUILT on the CPU even when cfg.device='cuda', and
    # train_model moves it for the duration of fit() only.
    #
    # It used to be built directly on the device, which quietly made everything between
    # build_model and train_model GPU code. It is not: gantry_interconnect_dynamic.py:339-340
    # calls encoder_init_state, which constructs its inputs with torch.tensor(...) on the CPU and
    # feeds them to fit_sys.encoder. On a CUDA build that raises inside pre_encoder.py at
    # `uhist_mod + self.u_off`. The same argument applies after fit() returns, which is why the
    # matching .cpu() at the end of train_model exists.
    #
    # Three call sites touch fit_sys before training (build_closed_loop, encoder_init_state x2)
    # and none of them wants a GPU, so the transition belongs around fit(), not around the build.
    # Moving after the optimizer is constructed is safe: nn.Module._apply rebinds param.data in
    # place, so the optimizer's parameter references stay valid, and Adam's state is allocated
    # lazily on the first step, by then on the right device.
    #
    # Availability is still checked HERE rather than in RunConfig.__post_init__: a config is
    # constructed for inspection on machines without a GPU, but a run that asked for one and
    # silently got the CPU would be 30x slower and its config.json would claim otherwise. Checking
    # at build time also fails the run before the window array is built, not after.
    if cfg.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'device="cuda" was requested but torch.cuda.is_available() is False. Falling back to '
            'the CPU is deliberately NOT done: the run would be slower by more than an order of '
            'magnitude while config.json recorded DEVICE=cuda, and the two would be compared as '
            'if they were the same experiment. Check the SLURM allocation (--gres=gpu:1) or set '
            'device="cpu" explicitly.')
    fit_sys.init_model(sys_data=data.train_data, device='cpu', auto_fit_norm=False,
                       optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps})
    eps_values = {float(group['eps']) for group in fit_sys.optimizer.param_groups}
    if eps_values != {float(cfg.adam_eps)}:
        raise RuntimeError(
            f'Adam eps mismatch after optimizer construction: config={cfg.adam_eps}, '
            f'optimizer={sorted(eps_values)}')
    if cfg.encoder_init in ('linear_map', 'data_lsq'):
        fit_sys.hfn.to(DTYPE_PT)
    else:
        for net in (fit_sys.encoder, fit_sys.hfn):
            net.to(DTYPE_PT)

    # Orthogonal-projection penalty (D7.1/D7.4): attached when cfg.orth is set.
    # fit_sys.orth_penalty stays None otherwise, which is the exact no-op of D7.2.
    # CHANGED (2026-08-28): gated on the boolean `cfg.orth`, not on `cfg.orth_beta > 0`.
    # __post_init__ guarantees beta > 0 whenever orth is True, so the attached penalty is
    # always a real one; the observe branch (attach with beta possibly 0) is gone with the
    # field. `orth=False` also skips the basis build, which the observe mode always paid for.
    if cfg.orth:
        from common.orth_penalty import build_orth_penalty
        fit_sys.orth_penalty = build_orth_penalty(cfg, data, norm)

    # One-step orthogonal-by-construction (D-192), attached the same way as the penalty above and
    # for the same reason: `obc=False` attaches nothing, so every seam it uses is inert and the
    # run is bit-identical to one from before it existed. The attach builds the fixed reference
    # set from every training sample (CPU, float64 source) and the correction module; the first
    # basis is built at the first objective evaluation, on the training device.
    if cfg.obc:
        from .obc_gantry import attach_obc
        attach_obc(fit_sys, cfg, data, norm)

    return fit_sys


def prefit_physical_encoder(fit_sys, cfg: RunConfig, data, norm):
    """D-251, D-271: start the PHYSICAL rows of the encoder at the least-squares map of a closed-loop
    baseline simulation.

    The model at its start (theta^0, f_aug = 0) is simulated in closed loop over every training
    record, from the measured positions with zero velocities, zero added states and zero controller
    state. A temporary callable around `fit_sys.hfn` records the state before each step and the
    corrected input of that step; `fit_sys.hfn` itself is never replaced and the rollout is the
    pipeline's own. The windows (y_hat, u_cl) of the na samples before tau and the simulated physical
    state at tau then give the least-squares map of encoder paper Eq. (33) (`least_squares_start`);
    the hidden layers, W^a and the added-state rows are left unchanged.
    """
    # Runs on the run's device, as fit() does (model.py::train_model): the rollout over every
    # training record is the cost, minutes on the GPU against about 25 min on four CPU cores. The
    # model goes back to the CPU afterwards, where build_model left it.
    if cfg.device == 'cuda':
        fit_sys.cuda()
    try:
        _prefit(fit_sys, cfg, data, norm)
    finally:
        if cfg.device == 'cuda':
            fit_sys.cpu()


def _prefit(fit_sys, cfg, data, norm):
    from model_augmentation.fit_systems.closed_loop import closed_loop_rollout
    enc, sim = fit_sys.encoder, fit_sys.simulator
    if sim is None:
        raise RuntimeError('the encoder pre-fit needs the closed loop: call build_closed_loop first')
    nx, na, nb, stride = cfg.nx_phys, fit_sys.na, fit_sys.nb, cfg.stride
    dtype, dev = next(enc.parameters()).dtype, next(enc.parameters()).device
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=dev)   # noqa: E731
    rv = lambda a: np.asarray(a, np.float64).ravel()                      # noqa: E731
    # With OBC attached the correction needs a coefficient paired with theta^0; it is zero here
    # (f_aug = 0 on the physical rows), and the basis built now is the one fit() would build first.
    fit_sys._sync_prediction_state_for_validation()
    # The parameter-dependent structures are constant over the rollout, so they are built once
    # and threaded, exactly as SSE_Interconnect_Composed.loss does (D-194, D-195).
    step_kw = {}
    with torch.no_grad():
        if fit_sys._mats_ix is not None:
            step_kw['pass_mats'] = fit_sys.hfn.connected_blocks[fit_sys._mats_ix].pass_mats()
        corr = getattr(fit_sys.hfn, 'obc_correction', None)
        if corr is not None:
            step_kw['obc_pass'] = corr.tangent_pair(corr.theta_frozen)

    sdl = data.train_data.sdl
    PinvT = np.linalg.inv(P.detach().cpu().numpy().astype(np.float64).T)
    x_mean, std_x = rv(norm.x_mean), rv(norm.std_x)
    uh_all, yh_all, xt_all = [], [], []
    groups = {}
    for i, sd in enumerate(sdl):
        groups.setdefault(len(sd.u), []).append(i)
    for ixs in groups.values():
        un = np.stack([(sdl[i].u - rv(fit_sys.norm.u0)) / rv(fit_sys.norm.ustd) for i in ixs])
        yn = np.stack([(sdl[i].y - rv(fit_sys.norm.y0)) / rv(fit_sys.norm.ystd) for i in ixs])
        x0 = np.zeros((len(ixs), nx + cfg.nx_ann))
        q0 = np.stack([np.asarray(sdl[i].y[0], np.float64) @ PinvT.T for i in ixs])  # q = P^-T y
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
    """D-271: set the physical rows of the encoder to the least-squares map x ~ W [yh; uh] + b.

    Encoder paper Eq. (33) with an intercept, on centred and per-column scaled regressors, solved in float64
    on the CPU by LAPACK gelsd (SVD) at the PyTorch default cutoff max(N, m) eps (a numerical convention):
    the numerical-rank-truncated SVD solution in standardised regressors. W^b takes W; the physical output
    rows of psi_tilde get zero weights and a bias carrying the intercept (minus the encoder's own constant,
    the D-055 offsets), so the physical part of the encoder is exactly the affine map. W^a and the
    added-state rows are left unchanged. Stops the run (RuntimeError) if any value is not finite, if the
    encoder does not reproduce the solved predictions, or if a state is reconstructed with R^2 < 0.999.
    """
    final = enc.net[-1]
    wa_before = (enc.Wa_psi_y.detach().clone(), enc.Wa_psi_u.detach().clone())
    cpu = lambda a: a.detach().to('cpu', torch.float64)                   # noqa: E731
    Phi, X = cpu(torch.cat([yh.flatten(1), uh.flatten(1)], 1)), cpu(xt)  # the encoder's own input order
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
    sv = sol.singular_values
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
          % (Phi.shape[1], int(sol.rank), float(sv[0] / sv[-1]), ' '.join('%.2e' % v for v in unexplained)),
          flush=True)
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
    # CHANGED (D-169, job 80714): the ONE place the model goes to the GPU, paired with the .cpu()
    # after fit() returns. Everything outside this pair -- the pre-training encoder-init capture,
    # the post-training baselines and the ~20 diagnostics -- is CPU and NumPy by construction, and
    # stays that way without a single device-aware branch. Validation runs INSIDE fit(), so it
    # happens on the GPU: measured 162 s there (80713) against the ~6 min this file's older
    # comment records for the CPU, because the free run is batched over the 4 records.
    if cfg.device == 'cuda':
        fit_sys.cuda()
    fit_sys.fit(
        train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
        batch_size=hp['batch_size'], epochs=epochs or hp['epochs'],
        # CHANGED: expose fit's existing n_its cap (interconnect.py:778). None keeps the
        # epochs-derived value, i.e. an exact no-op for every run that does not set it.
        n_its=cfg.n_its,
        # its_per_val defaults to 'epoch', which fit resolves to N_batch_updates_per_epoch
        # (260 here), so a capped run of e.g. 40 updates would never validate and report
        # nothing. Validate ONCE, at the cap. Not more often: one validation is a closed-loop
        # free run over the whole val set (4 records x 48000 samples, ~192k sequential
        # single-threaded steps, measured ~6 min), so it costs more than the 40 updates it
        # would be measuring. Start-and-end is what a smoke test needs. Untouched (and
        # therefore 'epoch') whenever n_its is None.
        # CHANGED (D-169): cadence now configurable. None keeps 'epoch', the historical default,
        # so this is an exact no-op unless cfg.its_per_val is set. n_its keeps precedence for the
        # reason above: a capped smoke test must validate at its cap, not never.
        its_per_val=(cfg.n_its if cfg.n_its else (cfg.its_per_val or 'epoch')),
        auto_fit_norm=False,
        loss_kwargs={'nf': nf if nf is not None else hp['nf'], 'stride': cfg.stride},
        optimizer_kwargs={'lr': hp['lr'], 'eps': cfg.adam_eps},
        # CHANGED (D-169): fit's own device flag. It drives two things init_model cannot: the
        # per-batch move (interconnect.py:823) and the cpu()/cuda() flip around each validation
        # (:716,:734). Without it a CUDA model would be fed CPU batches on the first update.
        cuda=(cfg.device == 'cuda'),
        validation_measure=validation_measure if validation_measure is not None else 'sim-RMS',
    )
    # CHANGED (D-169, job 80708): the other half of the pair above. Post-training evaluation and
    # the diagnostics are eager and mix in NumPy arrays, and every one of them was written when a
    # CPU model was guaranteed: fit() used to flip to the CPU around each validation and simply
    # leave it there. That guarantee is gone now that the flip is suppressed while a simulator is
    # attached (interconnect.py::fit), so state it once, here, rather than making twenty
    # diagnostics device-aware for no benefit. Free: training is over, and nothing downstream is
    # faster on the GPU.
    if cfg.device == 'cuda':
        fit_sys.cpu()
    return fit_sys.bestfit
