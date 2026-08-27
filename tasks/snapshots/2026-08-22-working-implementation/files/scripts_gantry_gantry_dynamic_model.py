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


class AugLRUBypass(torch.nn.Module):
    """Splits f_aug from g_aug (D-150): a stable linear bypass on the augmented output rows.

    Wraps the ANN's `zero_init_feed_forward_nn` so its output rows carry the two functions
    Hoekstra's S-DP structure keeps separate (arXiv:2602.17297 Table 1):

        rows 0-5 (f_aug)   unchanged: exactly zero at init, D-072 baseline equality intact
        rows 6-7 (g_aug)   x_a,k+1 = A_aa x_a,k + gamma * NL(x,u)[6:8], NL zero at init

    A_aa is held per complex-conjugate pair in the LRU stable exponential parameterisation
    # THEORY: Orvieto et al., ICML 2023 (Linear Recurrent Unit), Sec. 3.3: lambda =
    # exp(-exp(nu_log) + j exp(theta_log)) keeps abs(lambda) < 1 during training by construction,
    # and Sec. 3.4: the input into the state is normalised by gamma = sqrt(1 - abs(lambda)^2) so a
    # near-unit-circle initialisation does not blow up the forward pass.
    The real realisation of one pair is the 2x2 rotation-scaling block r*[[cos w, -sin w],
    [sin w, cos w]], whose spectral radius is r exactly and independently of state. That
    state-independence is the measured argument for a bypass: a tanh MLP output row asked for
    rho 0.98 at 159 Hz delivered 0.649 at 38.5 Hz (RESULTS.md section 4, 2026-08-19).

    `aug_out_pos` are the columns of the ANN OUTPUT that write the augmented state rows (the
    positions of those rows inside cfg.ann_route_ix); `x_aug_in_pos` are the columns of the ANN
    INPUT z = [x, u] that hold x_a. Both are plain tuples so the module pickles like anything
    else (deepSI checkpoints the whole __dict__; see rezero_gate.py for the precedent).

    The inner MLP is stored as `self.mlp`; `self.net` is a property returning the MLP's
    nn.Sequential so existing consumers (`ann.net.net[0]`, `apply_rezero_gate(ann_block.net)`)
    keep working unchanged.
    """

    def __init__(self, mlp, aug_out_pos, x_aug_in_pos, r_init, theta_init,
                 B_a=None):
        super().__init__()
        assert len(aug_out_pos) == len(x_aug_in_pos)
        assert len(aug_out_pos) % 2 == 0, 'augmented states come in complex-conjugate pairs'
        self.mlp = mlp
        # D-151: the input path into the augmented states. None reproduces D-150 exactly.
        # Trainable, like the linear part of Hoekstra's ResNet learning function (arXiv:2602.17297
        # Sec. 5.4.3, phi_aug(z_a) = 0 + W_a z_a); it gets zero gradient until the readout leaves
        # zero, by the same D-072 mechanism that holds nu_log/theta_log at their initialisation.
        self.B_a = None if B_a is None else torch.nn.Parameter(B_a)
        self.aug_out_pos = tuple(int(i) for i in aug_out_pos)
        self.x_aug_in_pos = tuple(int(i) for i in x_aug_in_pos)
        r = np.asarray(r_init, dtype=np.float64)
        th = np.asarray(theta_init, dtype=np.float64)
        assert r.shape == th.shape == (len(aug_out_pos) // 2,)
        assert np.all((r > 0) & (r < 1)) and np.all(th > 0), (
            'the exponential parameterisation needs 0 < r < 1 and theta > 0')
        # THEORY: Orvieto et al. ICML 2023 Sec. 3.3 -- nu_log = log(-log r), theta_log = log theta.
        self.nu_log = torch.nn.Parameter(torch.as_tensor(np.log(-np.log(r))))
        self.theta_log = torch.nn.Parameter(torch.as_tensor(np.log(th)))

    @property
    def net(self):
        return self.mlp.net

    def forward(self, X):
        w = self.mlp(X)
        r = torch.exp(-torch.exp(self.nu_log))            # (n_pairs,), abs(lambda) < 1 always
        th = torch.exp(self.theta_log)                    # (n_pairs,), rad/sample
        # THEORY: Orvieto et al. ICML 2023 Sec. 3.4 -- gamma = sqrt(1 - abs(lambda)^2).
        gamma = torch.sqrt(1.0 - r * r)
        xa = X[..., list(self.x_aug_in_pos)]              # (..., 2*n_pairs)
        x1, x2 = xa[..., 0::2], xa[..., 1::2]
        c, s = torch.cos(th), torch.sin(th)
        g = w[..., list(self.aug_out_pos)]                # NL part, zero at init
        if self.B_a is not None:
            # D-151: the input path, live at init. y reads x_a only through ANN rows 0-5, which
            # are exactly zero here, so this cannot move the output at t=0 and D-072 holds.
            g = g + X @ self.B_a.T
        g1, g2 = g[..., 0::2], g[..., 1::2]
        out = w.clone()
        out[..., list(self.aug_out_pos[0::2])] = r * (c * x1 - s * x2) + gamma * g1
        out[..., list(self.aug_out_pos[1::2])] = r * (s * x1 + c * x2) + gamma * g2
        return out


def lru_band_from_artifact(artifact_path, ts):
    """(f_band_hz, rho_band) from a cl_residual_spectrum.py artefact: a BAND, not a mode (D-150).

    Mirrors the artefact producer's READING 1 exactly: per record-channel, the dominant peak
    among those with a trustworthy half-power zeta
    # HEURISTIC: over_floor_db > 10 -- the same strong-peak threshold cl_residual_spectrum.py
    # READING 1 uses; a chosen threshold, not derived.
    The band is the [min, max] of those dominant frequencies across all records and channels,
    i.e. the estimator scatter itself is the ring width, and the radius band is the [min, max] of
    the per-peak pole magnitudes
    # THEORY: discrete pole magnitude of a second-order mode, rho = exp(-zeta*wn*Ts) with
    # wn = 2*pi*f / sqrt(1 - zeta^2) (matches cl_residual_spectrum.py READING 1).
    Raises with instructions when no strong peak exists: on such a dataset (the Telica case) the
    band must be supplied explicitly from loop-bandwidth and sample-rate requirements via
    AUG_LRU_BAND / AUG_LRU_RHO rather than silently defaulted here.
    """
    import json
    with open(artifact_path) as fh:
        art = json.load(fh)
    fz = []
    for rec in art['records'].values():
        for ch_modes in rec['modes']:
            strong = [m for m in ch_modes if m.get('zeta_ok') and m['over_floor_db'] > 10.0]
            if strong:
                best = max(strong, key=lambda m: m['over_floor_db'])
                fz.append((best['f_hz'], best['zeta']))
    if not fz:
        raise RuntimeError(
            'AUG_LRU: no strong residual peak in %s. On this dataset the initialisation band '
            'must come from band requirements (loop bandwidth, sample rate): set '
            'AUG_LRU_BAND="f_lo,f_hi" [Hz] and AUG_LRU_RHO="r_min,r_max" explicitly.'
            % artifact_path)
    f = np.array([q[0] for q in fz])
    z = np.array([q[1] for q in fz])
    wn = 2 * np.pi * f / np.sqrt(np.maximum(1 - z ** 2, 1e-12))
    rho = np.exp(-z * wn * ts)
    return (float(f.min()), float(f.max())), (float(rho.min()), float(rho.max()))


def get_encoder_dims(hp, cfg: RunConfig):
    """Return (na, nb, na_right, nb_right) consistent with build_model logic.

    AUG_LRU_NA_NB=<int> pins the encoder lag independently of nx_aug. Default unset, in which case
    this function is byte-identical to the pre-2026-08-21 behaviour.

    WHY THIS GATE EXISTS (added 2026-08-21, overnight diagnosis).
    Jan's rule na = nb = (nx_phys + nx_ann)*2 + 1 ties the encoder lag to the augmented-state count,
    so an nx_aug sweep silently sweeps the lag too: nx_aug 2, 8, 14 gives na_nb 17, 29, 41. A
    capacity result measured that way cannot be attributed to capacity.
    That was already a confound. Gate C1 turned it into a blocker: the D-072 free-run gate holds
    BIT-IDENTICALLY only at na_nb = 17, and fails monotonically as the lag grows,
        na_nb  17 -> 0.000e+00 | 32 -> 1.336e-04 | 64 -> 2.775e-04 | 103 -> 4.028e-04
        (runs/d072_matrix_probe.json)
    because W^b = A^n O_n^{-1} is exact in exact arithmetic but the observability inverse is
    increasingly ill-conditioned in float32. So ANY nx_aug > 2 arm run under Jan's rule starts from
    a model that is not the baseline, and its result is uninterpretable for a second, worse reason.
    Pinning the lag at 17 is therefore not a convenience: it is what makes a capacity arm legal.

    # THEORY: Beintema, Schoukens, Toth 2023, Automatica 156:111210, Sec. 3.4 p.5 -- na and nb are
    # free design variables bounded below by the reconstruction order, not functions of the state
    # count; the rule nxd*2+1 is Jan's convention, and the method's own authors treat n as a knob.
    """
    NX_PHYS = cfg.nx_phys
    _default = 2 * (NX_PHYS + hp['NX_ANN']) + 1        # THEORY: nxd*2+1 (Jan's standard)
    # PRECEDENCE: an explicit cfg.na_nb_override always wins, so every existing sweep and every
    # dataclasses.replace(..., na_nb_override=...) call site is unaffected by this gate. The env pin
    # is only consulted when no override was set.
    _pin = os.environ.get('AUG_LRU_NA_NB')
    if getattr(cfg, 'na_nb_override', None) is None and _pin:
        na = int(_pin)
    else:
        na = hp.get('na_nb', _default)
    nb = na
    if cfg.encoder_init == 'linear_map':
        na_right = 1   # reconstructability map uses y(k), so window is [k-na, k]
        nb_right = 1
    else:
        na_right = 0
        nb_right = 0
    return na, nb, na_right, nb_right


def find_log_params(fit_sys):
    """The Parameterized_Gantry_State_Block's log_params tensor, or None.

    Returns None when joint_estimation is off (the block is the non-parameterized
    Gantry_State_Block, which has no log_params).
    """
    for m in fit_sys.hfn.connected_blocks:
        if isinstance(m, Parameterized_Gantry_State_Block):
            return m.log_params
    return None


def split_param_group(fit_sys, lr_theta, lr_rest, eps_theta=None):
    """P1: rebuild fit_sys.optimizer with log_params in its OWN group (@added).

    Called immediately after init_model, so there is no optimizer state to carry
    over. log_params must be REMOVED from the bulk group: it is an nn.Parameter of
    a submodule of hfn, so hfn.parameters() already yields it, and PyTorch raises
    on a parameter appearing in two groups.

    eps_theta (P1-e) overrides Adam's eps for the theta group ONLY. The bulk group
    keeps the optimizer's own eps, so the ANN/encoder side is untouched by this
    function and stays owned by whoever tunes it. None = inherit, no override.

    Returns the number of tensors in the bulk group, for the verification harness.
    """
    theta = find_log_params(fit_sys)
    if theta is None:
        raise RuntimeError(
            'cfg.lr_theta is set but no Parameterized_Gantry_State_Block was '
            'found: log_params only exists when cfg.joint_estimation is True.')
    rest = [p for g in fit_sys.optimizer.param_groups for p in g['params']
            if p is not theta]
    defaults = dict(fit_sys.optimizer.defaults)
    defaults.pop('lr', None)
    g_theta = {'params': [theta], 'lr': lr_theta}
    if eps_theta is not None:
        g_theta['eps'] = eps_theta
    fit_sys.optimizer = fit_sys.optimizer.__class__(
        [{'params': rest, 'lr': lr_rest}, g_theta], **defaults)
    return len(rest)


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
    if os.environ.get('AUG_LRU_NA_NB') and cfg.na_nb_override is None:
        # Recorded in every gated run's log, because the whole point of the pin is that the arm's
        # result can be attributed, and the lag is otherwise inferred from nx_aug by Jan's rule.
        print('[aug-lag] AUG_LRU_NA_NB pin ACTIVE: na = nb = %d (Jan\'s rule would give %d at '
              'nx_aug = %d). D-072 holds bit-identically only at 17 (gate C1, '
              'runs/d072_matrix_probe.json), so a pin away from 17 breaks baseline equality and the '
              'arm must record that it did.' % (na, 2 * (NX_PHYS + NX_ANN) + 1, NX_ANN))

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
    # ANN_REZERO_GATE=1 gives the original SHARED SCALAR gate; ANN_REZERO_GATE=row gives one gain
    # per output row. The per-row form exists because the eight corrections span nine decades in
    # normalised units (cl_capability.py), and regressing this same architecture onto the exact
    # target fits the absorber rows to 1-R^2 = 9.6e-05 with a per-row scale and fails them
    # completely (0.98) without one. Zero-init either way, so the augmented model still equals the
    # baseline at initialisation.
    _gate_mode = (os.environ.get('ANN_REZERO_GATE') or '').lower()
    if _gate_mode:
        from .rezero_gate import apply_rezero_gate
        _per_row = _gate_mode in ('row', 'per_row', 'perrow')
        # as_module=True: the parametrization form cannot be pickled and deepSI checkpoints the
        # whole __dict__ at every validation, so it would kill any real run at its first checkpoint.
        _alpha = apply_rezero_gate(ann_block.net, per_row=_per_row, as_module=True)
        print(f"[rezero] final layer re-initialised + zero "
              f"{'per-row' if _per_row else 'scalar'} gate "
              f"(alpha shape {tuple(_alpha.shape)}, max {float(_alpha.abs().max()):.1f}); "
              f"ANN output is still exactly zero at init")

    # AUG_LRU=1 (D-150): split f_aug from g_aug. Rows 0-5 of the ANN output stay exactly zero at
    # initialisation (D-072 baseline equality, structural); the augmented rows get a live stable
    # linear term x_a,k+1 = A_aa x_a + gamma*NL, ring-initialised over a data-derived BAND read
    # from the residual-spectrum artefact (never a hard-coded mode: the real Telica data cannot
    # supply one, handoff 2026-08-19 section 4.9). AUG_LRU_BAND/AUG_LRU_RHO override the artefact
    # for datasets without an identifiable peak. Default OFF so every existing run reproduces.
    if os.environ.get('AUG_LRU') and NX_ANN > 0:
        aug_state_ix = list(range(NX_PHYS, NX_PHYS + NX_ANN))          # state rows 6..7
        assert all(j in route_ix for j in aug_state_ix), (
            'AUG_LRU needs the augmented rows %s inside cfg.ann_route_ix %s'
            % (aug_state_ix, tuple(route_ix)))
        aug_out_pos = [int(np.where(route_ix == j)[0][0]) for j in aug_state_ix]
        _band_env, _rho_env = os.environ.get('AUG_LRU_BAND'), os.environ.get('AUG_LRU_RHO')
        if _band_env or _rho_env:
            assert _band_env and _rho_env, 'AUG_LRU_BAND and AUG_LRU_RHO must be set together'
            f_band = tuple(float(v) for v in _band_env.split(','))
            rho_band = tuple(float(v) for v in _rho_env.split(','))
            _band_src = 'explicit env override'
        else:
            _art = os.environ.get('AUG_LRU_ARTIFACT') or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '..', 'closed-loop-controller',
                'runs', 'cl_residual_spectrum.json')
            f_band, rho_band = lru_band_from_artifact(_art, TS_NEW)
            _band_src = os.path.normpath(_art)
        n_pairs = NX_ANN // 2
        assert 2 * n_pairs == NX_ANN, 'AUG_LRU pairs augmented states; NX_ANN must be even'
        # Dedicated generator: drawing from the global torch stream would shift every later
        # random init (encoder W^a) against previous runs of the same seed.
        _gen = torch.Generator().manual_seed(int(cfg.seed) + 150)
        _u = torch.rand(n_pairs, generator=_gen, dtype=torch.float64).numpy()
        _v = torch.rand(n_pairs, generator=_gen, dtype=torch.float64).numpy()
        # THEORY: Orvieto et al. ICML 2023 Lemma 3.2 -- uniform on the annulus [r_min, r_max]:
        # r = sqrt(u*(r_max^2 - r_min^2) + r_min^2); phase uniform over the covered range.
        r_init = np.sqrt(_u * (rho_band[1] ** 2 - rho_band[0] ** 2) + rho_band[0] ** 2)
        theta_lo, theta_hi = (2 * np.pi * f_band[0] * TS_NEW, 2 * np.pi * f_band[1] * TS_NEW)
        theta_init = theta_lo + _v * (theta_hi - theta_lo)
        # D-151: AUG_LRU_B=<scale> restores the input path into the augmented states. Default off,
        # so an AUG_LRU=1 build without it reproduces D-150 exactly.
        _b_env = os.environ.get('AUG_LRU_B')
        _B_a = None
        if _b_env:
            _b_scale = float(_b_env)
            _nz = nxd + nu
            # THEORY: Orvieto et al., ICML 2023 (LRU) Sec. 3.3 -- the input matrix of a stable
            # exponential-parameterised recurrence is drawn i.i.d. normal with variance 1/n_in, so
            # the driven state magnitude does not scale with the input dimension.
            _B_a = torch.randn(NX_ANN, _nz, generator=_gen, dtype=DTYPE_PT) / np.sqrt(_nz)
            # The augmented columns must stay zero: a non-zero entry there would feed x_a back into
            # x_a and move the pole off the band-initialised A_aa, voiding the LRU guarantee.
            _B_a[:, list(aug_state_ix)] = 0.0
            _B_a = _B_a * _b_scale
        ann_block.net = AugLRUBypass(ann_block.net, aug_out_pos, aug_state_ix, r_init, theta_init,
                                     B_a=_B_a)
        # AUG_LRU_FREEZE=1 (2026-08-22, factor run F5): hold nu_log and theta_log at their drawn
        # values for the whole run, so A_aa is a FIXED basis of poles spanning the band rather than
        # a trainable one. The arm exists because the trained pole tables move by under 0.15 Hz in
        # both arm 1 and arm 2 (runs/readout_jacobian_*.json), which is what a fixed basis looks
        # like; if freezing costs nothing, the poles were never being learned and the mechanism is
        # "a spanning set of resonators", not "a mode that is identified".
        # Default OFF, so an unset build is byte-identical to every earlier run.
        if os.environ.get('AUG_LRU_FREEZE'):
            ann_block.net.nu_log.requires_grad_(False)
            ann_block.net.theta_log.requires_grad_(False)
            print('[aug-lru-freeze] F5: nu_log and theta_log frozen at their drawn values '
                  '(requires_grad False). B_a and the MLP stay trainable.')
        if _B_a is not None:
            print('[aug-lru-b] D-151 input injection ON: B_a %s, scale %.4g, augmented columns %s '
                  'zeroed, ||B_a||_F = %.4e; rows 0-5 still exactly zero at init (D-072 intact)'
                  % (tuple(_B_a.shape), _b_scale, tuple(aug_state_ix), float(_B_a.norm())))
        print('[aug-lru] D-150 bypass on augmented rows %s: band f [%.2f, %.2f] Hz, '
              'rho [%.4f, %.4f] (source: %s); drawn lambda: %s; rows 0-5 still exactly zero at init'
              % (aug_state_ix, f_band[0], f_band[1], rho_band[0], rho_band[1], _band_src,
                 '  '.join('r %.4f at %.2f Hz' % (r_init[p], theta_init[p] / (2 * np.pi * TS_NEW))
                           for p in range(n_pairs))))

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
        # ENC_WA_ZERO=1: start the AUGMENTED encoder rows at zero instead of the kaiming-uniform
        # random map. Neither is Jan's: hoekstra2026encoder defines W^a in Eq. 8 but initialises
        # only W^b (Eqs. 16-17, 28-29, 31-32, 33-35, all from the BASELINE's reconstructability,
        # which has no augmented states to reconstruct), and its own experiment is a static
        # augmentation with nx_aug = 0, so W^a appears in no result in the paper. See
        # `pre_encoder.linear_encoder_init_aug` and docs/references.md.
        # MEASURED (2026-08-19, cl_capability.py, planted-model window RMS on V1-V4):
        #   W^a random (today) 1.2068e-06 | W^a = 0  7.6156e-07 | true latent x0  7.1603e-07
        # i.e. zeroing it is worth 1.59x on the objective the loss minimises and recovers 90.7 % of
        # what a perfect latent observer would buy. It also makes this encoder agree with the
        # repo's other two (HybridGantryEncoder, LinearInitEncoderWrapper), which already take
        # their augmented states from a zero-init ANN. Default OFF: this is a stated design change
        # to be measured per run, not a silent switch.
        if os.environ.get('ENC_WA_ZERO') and NX_ANN > 0:
            with torch.no_grad():
                fit_sys.encoder.Wa_psi_y.zero_()
                fit_sys.encoder.Wa_psi_u.zero_()
            print('[encoder] W^a zeroed (Wa_psi_y, Wa_psi_u): the augmented rows start at 0 '
                  'instead of a random linear map of the history')
        # ENC_WA_FREEZE=1 (2026-08-22, factor run F3c): additionally hold W^a at its initial value,
        # so the encoder's augmented block is not a free parameter at all. Combined with
        # ENC_WA_ZERO=1 this is behaviourally the "no W^a block" case, i.e. it is how
        # pre_encoder.linear_encoder_init_aug's W^a addition gets ablated without editing
        # model_augmentation/. Separate from ENC_WA_ZERO because that gate sets only the initial
        # VALUE, and T1's derivation (Hoekstra Eq. (7), the encoder approximates E[x_a | u,y], which
        # is zero when the readout is exactly zero under D-072) says nothing about trainability.
        # Default OFF; unset, this block does not execute.
        if os.environ.get('ENC_WA_FREEZE') and NX_ANN > 0:
            fit_sys.encoder.Wa_psi_y.requires_grad_(False)
            fit_sys.encoder.Wa_psi_u.requires_grad_(False)
            print('[encoder] W^a FROZEN (Wa_psi_y, Wa_psi_u requires_grad False): the augmented '
                  'encoder block is held at its initial value for the whole run')

    # CHANGED: pass lr at optimizer creation. init_model builds the optimizer here;
    # fit()'s optimizer_kwargs are ignored once init_model_done=True, so without this
    # every run trained at Adam's default 1e-3 instead of hp['lr'] (D-101).
    fit_sys.init_model(sys_data=data.train_data, auto_fit_norm=False,
                       optimizer_kwargs={'lr': hp['lr']})
    # CHANGED: P1 -- give log_params its own optimizer group at cfg.lr_theta.
    # deepSI cannot express this via parameters_optimizer_kwargs: that dict is
    # keyed by the names in parameters_with_names (fit_system.py:187-196), which
    # walks dir(self) for top-level nn.Module / nn.Parameter attributes, and
    # log_params is nested inside hfn. No-op when cfg.lr_theta is None.
    if cfg.lr_theta is not None:
        split_param_group(fit_sys, cfg.lr_theta, hp['lr'], cfg.eps_theta)
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
