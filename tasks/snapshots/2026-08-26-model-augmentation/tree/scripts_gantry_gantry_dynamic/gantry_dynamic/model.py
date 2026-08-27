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


def _augmented_drive_slice(phy_block, data, norm, cfg, nxd, nx_phys, dtype, n=2000):
    """The correction net's ACTUAL input `z = [x, u]` over a training slice, at initialisation.

    Used only to SCALE the augmented input map (Schoukens ECC 2021 Sec. IV: states normalised to
    unit standard deviation). At init the augmented model IS the baseline (D-072), so the physical
    part of `z` is exactly what the frozen physical block produces from the recorded input, which
    is why rolling that block forward is the right slice and no encoder or closed loop is needed.
    The augmented columns are left at zero: `B` masks them, so they cannot affect the drive.

    `n = 2000` is 0.5 s at the 4 kHz model rate, i.e. about 79 periods of the absorber mode, which
    is ample for a standard deviation. The recursion is sequential so it costs a few seconds; it
    runs once per build and never during training.
    """
    u = np.asarray(data.train_data.u, dtype=np.float64)[:n]
    # norm.u_mean / norm.std_u are stored as COLUMN vectors (nu, 1); ravel so they broadcast
    # against the (N, nu) record rather than transposing it
    u_n = (u - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
    Z = np.zeros((len(u_n), nxd + cfg.nu), dtype=np.float64)
    # the block's convention is (batch, nx + nu, 1); see Gantry_State_Block.nonlinear_function
    x = torch.zeros((1, nx_phys, 1), dtype=dtype)
    with torch.no_grad():
        for k in range(len(u_n)):
            uk = torch.as_tensor(u_n[k], dtype=dtype).reshape(1, -1, 1)
            Z[k, :nx_phys] = x.reshape(-1).cpu().numpy()
            Z[k, nxd:] = u_n[k]
            x = phy_block(torch.cat([x, uk], dim=1))
    return Z


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
    aug_state_method = cfg.resolved_aug_state_method

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
    legacy_ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=len(route_ix),  # D-068: ANN outputs, one per routed row (cfg.ann_route_ix)
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=_act,
    )
    ann_block = legacy_ann_block
    kessels_block = None
    physical_route_ix = route_ix
    physical_head_equivalence = None

    if aug_state_method == 'kessels_extension':
        from model_augmentation.fit_systems.kessels_extension import (
            KesselsExtensionBlock, split_physical_ann)
        aug_ix = np.arange(NX_PHYS, nxd)
        missing = [int(j) for j in aug_ix if j not in route_ix.tolist()]
        if missing:
            raise ValueError('kessels_extension requires every augmented row in ann_route_ix; '
                             f'missing {missing}')
        physical_route_ix = route_ix[route_ix < NX_PHYS]
        if not len(physical_route_ix):
            raise ValueError('kessels_extension requires at least one retained physical correction row')
        physical_positions = [int(np.where(route_ix == j)[0][0]) for j in physical_route_ix]
        ann_block, physical_head_equivalence = split_physical_ann(
            legacy_ann_block, physical_positions)
        if cfg.writer_initialization == 'kessels_small':
            # Declared liveness control: the existing project reader is not Kessels l_w1, so this
            # is intentionally labelled inspired rather than a full published initialization.
            _last = [m for m in ann_block.net.modules() if isinstance(m, torch.nn.Linear)][-1]
            torch.nn.init.normal_(_last.weight, mean=0.0, std=cfg.writer_small_init_std)
            torch.nn.init.normal_(_last.bias, mean=0.0, std=cfg.writer_small_init_std)

        n_in = nxd + nu
        multiplier = np.ones(n_in, dtype=np.float64)
        offset = np.zeros(n_in, dtype=np.float64)
        writer_scaling_evidence = {
            'mode': cfg.writer_io_scaling,
            'fit_records': list(getattr(__import__('gantry_dynamic.data', fromlist=['TRAIN_FILES']),
                                        'TRAIN_FILES')),
        }
        if cfg.writer_io_scaling == 'kessels_fp_range':
            x_range = np.ptp(np.asarray(norm.x_all, dtype=np.float64), axis=0)
            u_range = np.ptp(np.asarray(norm.u_all, dtype=np.float64), axis=0)
            if (not np.isfinite(x_range).all() or not np.isfinite(u_range).all()
                    or np.any(x_range == 0.0) or np.any(u_range == 0.0)):
                raise ValueError('kessels_fp_range refuses zero or nonfinite T1--T14 range')
            # Outer state/input coordinates are (raw-mean)/std.  This affine adapter yields
            # raw/range, i.e. Kessels' pure diagonal range scaling without centering.
            multiplier[:NX_PHYS] = np.asarray(norm.std_x).reshape(-1) / x_range
            offset[:NX_PHYS] = np.asarray(norm.x_mean).reshape(-1) / x_range
            multiplier[nxd:] = np.asarray(norm.std_u).reshape(-1) / u_range
            offset[nxd:] = np.asarray(norm.u_mean).reshape(-1) / u_range
            writer_scaling_evidence.update(
                x_range=x_range.tolist(), u_range=u_range.tolist(),
                equation='scaled=(outer_normalized*std+mean)/(max-min)',
                inverse_equation='outer_normalized=(scaled*(max-min)-mean)/std',
            )
        else:
            writer_scaling_evidence['equation'] = 'identity on repository normalized [x_b,x_a,u]'
            writer_scaling_evidence['inverse_equation'] = 'identity'

        kessels_block = KesselsExtensionBlock(
            nx_phys=NX_PHYS, n_aug=NX_ANN, nu=nu, sample_time=TS_NEW,
            coordinate_mode=cfg.xa_coordinate_mode,
            writer_io_scaling=cfg.writer_io_scaling,
            writer_input_multiplier=multiplier, writer_input_offset=offset,
            writer_linear_bypass=cfg.writer_linear_bypass,
            writer_hidden_widths=cfg.writer_hidden_widths,
            writer_initialization=cfg.writer_initialization,
            small_init_std=cfg.writer_small_init_std, dtype=DTYPE_PT,
        )

    ic.add_block(ann_block)
    if kessels_block is not None:
        ic.add_block(kessels_block)

    # ReZero-style gate: zero ANN OUTPUT at init WITHOUT a zero output PROJECTION, so the branch
    # F(z) is live and the gate alpha receives gradient at step 1 (arXiv:2003.04887; the degenerate
    # zero-projection-plus-zero-gate saddle is arXiv:2607.16568). Addresses the W^a dead zone
    # (D-130, gate G1).
    if aug_state_method == 'kessels_extension' and os.environ.get('ANN_REZERO_GATE'):
        raise ValueError(
            'ANN_REZERO_GATE is not a declared Kessels initialization mode; unset it and use '
            'writer_initialization=matched_project or kessels_small explicitly')
    if os.environ.get('ANN_REZERO_GATE'):
        from .rezero_gate import apply_rezero_gate
        _alpha = apply_rezero_gate(ann_block.net)
        print(f"[rezero] final layer re-initialised + zero scalar gate (alpha={float(_alpha):.1f}); "
              f"ANN output is still exactly zero at init")

    # CHANGED (D-160): install the augmented-state recurrence, so x_a is DRIVEN at init instead of
    # being identically zero. Without it nothing writes rows nx_phys.. (the readout's augmented
    # columns are zero and the ANN output is zero), the ANN's read weights on those columns get
    # zero gradient forever, and the states are measurably dead: ablation 1.0002x, F = 0.0007
    # (BLA-Augmentation/RESULTS.md:298). OFF by default, so every prior run reproduces.
    if aug_state_method == 'linear_ab':
        if NX_ANN <= 0:
            raise ValueError('aug_dynamics needs nx_ann > 0')
        _aug_ix = list(range(NX_PHYS, NX_PHYS + NX_ANN))
        _missing = [j for j in _aug_ix if j not in route_ix.tolist()]
        if _missing:
            raise ValueError('aug_dynamics requires the augmented rows to be routed; '
                             'ann_route_ix is missing %s' % _missing)
        if not len(cfg.aug_rho) or not len(cfg.aug_theta):
            # REFUSAL, not a default. The poles do not move during training (C6/T3; measured
            # motion under 0.15 Hz over 520 updates), so a silent default would silently decide
            # the result. D-160 requires them to come from an identification of the residual.
            raise ValueError(
                'aug_dynamics is on but aug_rho/aug_theta are empty. The augmented poles are '
                'frozen by the objective, so placing them decides the outcome and there is no '
                'defensible default; supply them from an identification (D-160).')
        from model_augmentation.fit_systems.augmented_dynamics import (
            OUTPUT_ZERO, AugmentedDynamics, draw_input_map, empirical_input_scale)
        _nz = nxd + nu
        _aug_out_pos = [int(np.where(route_ix == j)[0][0]) for j in _aug_ix]
        _B = draw_input_map(NX_ANN, _nz, _aug_ix, seed=cfg.seed + 161)
        # Scale B so x_a has unit std under the REAL driving signal, per Schoukens ECC 2021
        # Sec. IV ("normalized such that each of the states has a standard deviation equal to 1").
        # Z is built from the frozen physical block on a slice of training input, which is what the
        # correction net actually sees at init because the model IS the baseline there. Replaces
        # Orvieto's sqrt(1 - rho^2), whose white-input premise does not hold for this z.
        _Z = _augmented_drive_slice(phy_block, data, norm, cfg, nxd, NX_PHYS, DTYPE_PT)
        _B, _std0 = empirical_input_scale(_B, cfg.aug_rho, cfg.aug_theta, _Z)
        ann_block.net = AugmentedDynamics(
            ann_block.net, _aug_out_pos, _aug_ix,
            rho=cfg.aug_rho, theta=cfg.aug_theta, B=_B,
            gamma_on_B=False, gate_mode=OUTPUT_ZERO,
            train_A=cfg.aug_train_A, train_B=cfg.aug_train_B,
            dtype=DTYPE_PT).to(DTYPE_PT)
        print('[aug-dyn] input map scaled to unit x_a std; pre-scale std per pair %s'
              % np.array2string(_std0, precision=4))
        print('[aug-dyn] %d augmented states, %d pole pairs, rows %s -> ANN output cols %s'
              % (NX_ANN, NX_ANN // 2, _aug_ix, _aug_out_pos))
        for _r in ann_block.net.pole_table(TS_NEW):
            print('          pair %d: rho %.6f  f_d %.4f Hz  zeta %.5f'
                  % (_r['pair'], _r['rho'], _r['f_d_hz'], _r['zeta']))

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
    ic.connect_signals(ann_block, "xp", "additive", expansion_matrix(physical_route_ix, nxd))
    if kessels_block is not None:
        ic.connect_block_signals(kessels_block, ["x", "u"], [])
        ic.connect_signals(kessels_block, "xp", "additive",
                           expansion_matrix(np.arange(NX_PHYS, nxd), nxd))
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
    fit_sys.aug_state_method = aug_state_method
    fit_sys.physical_route_rows = tuple(int(i) for i in physical_route_ix)
    if kessels_block is not None:
        fit_sys.kessels_manifest = kessels_block.manifest()
        fit_sys.kessels_manifest.update({
            'physical_route_rows': list(fit_sys.physical_route_rows),
            'physical_route_departure': 'retained project ANN may write physical position rows',
            'controller_mode': cfg.controller_mode,
            'controller_fidelity': 'project residual replay; not exact native-Kessels data flow',
            'validation_records_role': cfg.validation_records_role,
            'physical_history_end': 'k0', 'augmented_history_end': 'k0-1',
            'augmented_history_signals': ['u', 'y'],
        })
        fit_sys.writer_scaling_evidence = writer_scaling_evidence
        fit_sys.physical_head_equivalence = physical_head_equivalence
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
        legacy_encoder = linear_encoder_init_aug(
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
        if kessels_block is None:
            fit_sys.encoder = legacy_encoder
        else:
            from model_augmentation.fit_systems.kessels_extension import (
                HoekstraPhysicalEncoder, KesselsAugmentedEncoder, KesselsCompositeEncoder)
            physical_encoder = HoekstraPhysicalEncoder(legacy_encoder).to(DTYPE_PT)
            with torch.no_grad():
                _up = torch.linspace(-1, 1, 5 * (nb + 1) * nu, dtype=DTYPE_PT).reshape(5, nb + 1, nu)
                _yp = torch.linspace(1, -1, 5 * (na + 1) * ny, dtype=DTYPE_PT).reshape(5, na + 1, ny)
                _old_xb = legacy_encoder(_up, _yp)[:, :NX_PHYS]
                _new_xb = physical_encoder(_up, _yp)
                _enc_eq = torch.equal(_old_xb, _new_xb)
                _enc_diff = float((_old_xb - _new_xb).abs().max())
            if not _enc_eq:
                raise RuntimeError('physical encoder split-copy equivalence failed')
            aug_history_length = cfg.aug_history_length or na
            augmented_encoder = KesselsAugmentedEncoder(
                nu=nu, ny=ny, n_history=aug_history_length, n_aug=NX_ANN,
                initialization=cfg.writer_initialization,
                small_init_std=cfg.writer_small_init_std, dtype=DTYPE_PT,
            )
            fit_sys.encoder = KesselsCompositeEncoder(
                physical_encoder, augmented_encoder).to(DTYPE_PT)
            fit_sys.kessels_manifest['aug_history_length'] = aug_history_length
            fit_sys.kessels_manifest['physical_encoder_copy_equivalence'] = {
                'bitwise_equal': _enc_eq, 'max_abs_difference': _enc_diff}

    # CHANGED: pass lr at optimizer creation. init_model builds the optimizer here;
    # fit()'s optimizer_kwargs are ignored once init_model_done=True, so without this
    # every run trained at Adam's default 1e-3 instead of hp['lr'] (D-101).
    fit_sys.init_model(sys_data=data.train_data, auto_fit_norm=False,
                       optimizer_kwargs={'lr': hp['lr']})
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
    fit_sys.fit(
        train_sys_data=data.train_data, val_sys_data=data.val_ckpt_data,
        batch_size=hp['batch_size'], epochs=epochs or hp['epochs'],
        auto_fit_norm=False,
        loss_kwargs={'nf': nf if nf is not None else hp['nf'], 'stride': cfg.stride},
        optimizer_kwargs={'lr': hp['lr']},
        validation_measure=validation_measure if validation_measure is not None else 'sim-RMS',
    )
    return fit_sys.bestfit
