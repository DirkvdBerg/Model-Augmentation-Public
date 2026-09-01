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
    if cfg.encoder_init == 'linear_map':
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
