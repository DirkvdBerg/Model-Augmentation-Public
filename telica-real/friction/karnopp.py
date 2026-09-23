"""Karnopp (set-valued) Coulomb friction for the gantry baseline, batched torch (TR-006).

Port of the friction block of
`Matlab-scripts/Augmentation-coulomb/gantrySystemExtendedCoulomb.m` (lines 137-243), without the
hidden MSD. Friction is a STAGE-frame force per physical rail [X1, X2, Y], projected to logical
coordinates with P and subtracted from the input at BOTH u-sites of the LFR baseline:

    u_eff = u_log - P F_stage                          (D-116 / D-138 placement)

The law, per rail i:
    |v_i| >= V_BRK                 F_i = cc_i sign(v_i)                        slip (garcia2013)
    |v_i| <  V_BRK                 F_i solves a_stage,i = 0 through G          stuck, held
    stuck but |F_i| > cc_i         F_i = cc_i sign(F_i), rail leaves the stuck set  breakaway
with G = P' M(Y)^-1 P the stage-friction -> stage-acceleration map (Delassus matrix) and an
active set iterated 4 times (3 rails: exact).

`FrictionMixin` puts it into any `Gantry_State_Block` subclass (plain or `Reduced_...`) by
overriding `_deriv_with`; `make_friction_block` builds the common cases.
"""
import numpy as np
import torch
from torch import Tensor

from model_augmentation.fit_systems.blocks import Gantry_State_Block, Reduced_Gantry_State_Block
from params import telica_params as tp

MODES = ('karnopp', 'tanh', 'sign', 'none')


def karnopp_stage_force(v_stage: Tensor, a_free_stage: Tensor, G: Tensor, cc: Tensor,
                        v_brk: float, n_iter: int = 4, census: dict = None,
                        early_exit: bool = True) -> Tensor:
    """Set-valued Coulomb force per rail, STAGE frame, batched.

    v_stage       (B,3)   rail velocities [dX1, dX2, dY]
    a_free_stage  (B,3)   frictionless rail accelerations
    G             (B,3,3) stage friction -> stage acceleration (a = a_free - G F)
    cc            (1,3) or (B,3) Coulomb magnitudes [N]
    early_exit    True = stop when no rail breaks away (the .m file's loop, verified in G2).
                  False = always run n_iter passes: identical result (a pass with an unchanged
                  stuck set re-solves the same system), but no data-dependent Python branch, so
                  it compiles under torch.compile(fullgraph=True) (G6).
    Returns F_stage (B,3). `census` (optional dict) receives the final stuck / broke masks.
    """
    dtype = v_stage.dtype
    B = v_stage.shape[0]
    # gantrySystemExtendedCoulomb.m:183-186
    stuck = v_stage.abs() < v_brk
    # :197-203  THEORY: garcia2013 Fig. 2: slip force cc_i sign(v_i) on rails sliding from the start
    Fassign = torch.where(stuck, torch.zeros_like(v_stage), cc * torch.sign(v_stage))
    eye = torch.eye(3, dtype=dtype, device=v_stage.device).expand(B, 3, 3)
    stuck0 = stuck
    F = torch.zeros_like(v_stage)
    for _ in range(n_iter):                                        # :206
        m = stuck.unsqueeze(-1).to(dtype)
        Amat = m * G + (1 - m) * eye                               # :211-221
        bvec = torch.where(stuck, a_free_stage, Fassign).unsqueeze(-1)
        F = torch.linalg.solve(Amat, bvec).squeeze(-1)             # :222
        broke = stuck & (F.abs() > cc)                             # :228-234
        if early_exit and not bool(broke.any()):
            break
        Fassign = torch.where(broke, cc * torch.sign(F), Fassign)
        stuck = stuck & ~broke
    if census is not None:
        census['stuck0'] = stuck0
        census['held'] = stuck
        census['broke'] = stuck0 & ~stuck
    return F


class FrictionMixin:
    """Adds stage-frame Coulomb friction to a Gantry_State_Block subclass at both u-sites.

    Call `_init_friction(...)` after the block's own __init__. `cc` is a buffer (fixed) or, with
    `trainable_cc=True`, `cc_init * exp(log_cc)` (log coordinates keep it positive; G4 recovery).
    mode 'none' returns the parent's `_deriv_with` untouched.
    """

    def _init_friction(self, cc=tp.CC, mode='karnopp', v_brk=tp.V_BRK, v0=tp.V0_TANH,
                       trainable_cc=False):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {mode!r}')
        self.fric_mode = mode
        self.v_brk = float(v_brk)
        self.v0 = float(v0)
        dt = self.P_mat.dtype
        self.register_buffer('cc_init', torch.as_tensor(cc, dtype=dt).reshape(1, 3).clone())
        self.trainable_cc = bool(trainable_cc)
        if self.trainable_cc:
            self.log_cc = torch.nn.Parameter(torch.zeros(1, 3, dtype=dt))
        self.last_census = None
        self.record_census = False

    def cc_vec(self) -> Tensor:
        if self.trainable_cc:
            return self.cc_init * torch.exp(self.log_cc)
        return self.cc_init

    def _deriv_with(self, x: Tensor, u: Tensor, mats, terms: dict = None) -> Tensor:
        if self.fric_mode == 'none':
            return super()._deriv_with(x, u, mats, terms)
        # Restated from Gantry_State_Block._deriv_with (vendor blocks.py:810-871) TERM FOR TERM,
        # so cc = 0 is bit-identical to the parent (gate G2a). The only change: u_log -> u_eff
        # at both u-sites, u_eff = u_log - P F_stage.
        (K_mat, C_mat, A_combined, mh, alpha, beta, gamma_, N0, N1, N2) = mats
        x_phys = x * self.std_x + self.x_mean
        u_phys = u * self.std_u + self.u_mean
        x2 = x_phys.squeeze(-1)
        u2 = u_phys.squeeze(-1)
        u_log = u2 @ self.P_mat.T

        if self.Y_op is not None:
            Y_val = self.Y_op
        else:
            Y = x2[:, 2]
            dY = mh * (alpha * gamma_ - beta ** 2
                       + 2 * beta * mh * Y
                       + mh * (alpha - mh) * Y ** 2)
            Y_r = Y.unsqueeze(0)
            Y_val = Y[:, None]

        def _forward(u_eff):
            fnet = (-(x2[:, :3] @ K_mat.T)
                    - (x2[:, 3:] @ C_mat.T)
                    + u_eff)                                     # u-site 1
            if self.Y_op is not None:
                a = (self.N_op @ fnet.T).T / self.d_op
            else:
                n0f = N0 @ fnet.T
                n1f = N1 @ fnet.T
                n2f = N2 @ fnet.T
                a = (n0f + Y_r * (n1f + Y_r * n2f)).T / dY[:, None]
            z = torch.cat([a, Y_val * a], dim=-1)
            w = Y_val * z
            combined = torch.cat([x2, w, u_eff], dim=-1)         # u-site 2 (the only one at Y=0)
            return combined @ A_combined.T

        P = self.P_mat
        v_stage = x2[:, 3:] @ P                                  # P' qdot  (row convention)
        cc = self.cc_vec()
        if self.fric_mode == 'karnopp':
            a_free = _forward(u_log)[:, 3:]
            if self.Y_op is not None:
                Mr = (self.N_op / self.d_op).expand(x2.shape[0], 3, 3)
            else:
                Mr = ((N0 + Y[:, None, None] * N1 + (Y ** 2)[:, None, None] * N2)
                      / dY[:, None, None])
            G = P.T @ Mr @ P                                     # (B,3,3) Delassus, stage frame
            census = {} if self.record_census else None
            F = karnopp_stage_force(v_stage, a_free @ P, G, cc, self.v_brk, census=census,
                                    early_exit=getattr(self, 'early_exit', True))
            if census is not None:
                self.last_census = census
        elif self.fric_mode == 'tanh':
            F = cc * torch.tanh(v_stage / self.v0)               # D-116 surrogate, HEURISTIC v0
        else:
            F = cc * torch.sign(v_stage)                         # THEORY: garcia2013 Fig. 2
        u_eff = u_log - F @ P.T
        xdot_phys = _forward(u_eff)
        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        if terms is not None:
            terms.update(F_stage=F, u_eff=u_eff, xdot_phys=xdot_phys, xdot=xdot)
        return xdot


class GantryFrictionBlock(FrictionMixin, Gantry_State_Block):
    """Fixed-parameter baseline + friction."""

    def __init__(self, cc=tp.CC, mode='karnopp', v_brk=tp.V_BRK, v0=tp.V0_TANH,
                 trainable_cc=False, **kwargs):
        super().__init__(**kwargs)
        self._init_friction(cc, mode, v_brk, v0, trainable_cc)


class ReducedGantryFrictionBlock(FrictionMixin, Reduced_Gantry_State_Block):
    """Ten identifiable combinations trainable (D-190) + friction (optionally trainable cc)."""

    def __init__(self, cc=tp.CC, mode='karnopp', v_brk=tp.V_BRK, v0=tp.V0_TANH,
                 trainable_cc=False, **kwargs):
        super().__init__(**kwargs)
        self._init_friction(cc, mode, v_brk, v0, trainable_cc)


def lfr_constants64(p=None):
    """Every parameter-derived LFR constant in float64, from a dict of raw params (default: tp).

    The vendored gantry_ss builds its module constants in float32, i.e. ~1e-8 relative off the
    float64 values (P's Lb/2 = 0.3625 becomes 0.36250001...). Harmless on real data, fatal for a
    1e-12 m cross-check, so the blocks here are rebuilt from the float64 parameters.
    """
    from model_augmentation.systems.gantry_ss import build_poly_constants, build_G_matrix_entries
    D = torch.float64
    q = {n: getattr(tp, n) for n in ('m1', 'm2', 'mb', 'mh', 'Jb', 'Jh', 'Lb', 'd', 'cg1', 'cg2',
                                     'cy', 'cb1', 'cb2', 'kb1', 'kb2')}
    if p is not None:
        q.update(p)
    t = {k: torch.tensor(float(v), dtype=D) for k, v in q.items()}
    alpha, beta, gamma_, N0, N1, N2 = build_poly_constants(
        t['m1'], t['m2'], t['mb'], t['mh'], t['Jb'], t['Jh'], t['Lb'], t['d'])
    d0 = t['mh'] * (alpha * gamma_ - beta ** 2)
    z = torch.zeros(3, 3, dtype=D)
    M1 = z.clone(); M1[0, 1] = -t['mh']; M1[1, 0] = -t['mh']
    M2 = z.clone(); M2[1, 1] = t['mh']
    Lb = t['Lb']
    C = z.clone()
    C[0, 0] = t['cg1'] + t['cg2']
    C[0, 1] = C[1, 0] = (t['cg1'] - t['cg2']) * Lb / 2
    C[1, 1] = t['cb1'] + t['cb2'] + (t['cg1'] + t['cg2']) * Lb ** 2 / 4
    C[2, 2] = t['cy']
    K = z.clone(); K[1, 1] = t['kb1'] + t['kb2']
    P = z.clone(); P[0, 0] = 1.0; P[0, 1] = 1.0; P[1, 0] = Lb / 2; P[1, 1] = -Lb / 2; P[2, 2] = 1.0
    Ax, Bw, Bu, A_combined = build_G_matrix_entries(N0, d0, M1, M2, K, C)
    return dict(mh=t['mh'], alpha=alpha, beta=beta, gamma_=gamma_, N0=N0, N1=N1, N2=N2,
                Ax=Ax, Bw=Bw, Bu=Bu, A_combined=A_combined, K_mat=K, C_mat=C, P_mat=P, Lb=Lb)


def rebuild_float64(blk, p=None):
    """Cast `blk` to float64 and overwrite its LFR buffers with `lfr_constants64(p)`."""
    blk = blk.to(torch.float64)
    c = lfr_constants64(p)
    for name in ('mh', 'alpha', 'beta', 'gamma_', 'N0', 'N1', 'N2', 'Ax', 'Bw', 'Bu',
                 'A_combined', 'K_mat', 'C_mat', 'P_mat'):
        setattr(blk, name, c[name].clone())
    if 'Lb' in blk._buffers:
        blk.Lb = c['Lb'].clone()
    if blk.Y_op is not None:
        Yt = torch.tensor(float(blk.Y_op), dtype=torch.float64)
        mh, al, be, ga = c['mh'], c['alpha'], c['beta'], c['gamma_']
        blk.N_op = c['N0'] + c['N1'] * Yt + c['N2'] * Yt ** 2
        blk.d_op = mh * (al * ga - be ** 2 + 2 * be * mh * Yt + mh * (al - mh) * Yt ** 2)
    return blk


def _unit_norm():
    return dict(std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
                x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1)))


def make_friction_block(Y_op=None, cc=tp.CC, mode='karnopp', ts=tp.TS, up_sample=1,
                        dtype=torch.float64, **kw):
    """Unnormalised block (std 1, mean 0): physical units in and out, constants from float64."""
    blk = GantryFrictionBlock(cc=cc, mode=mode, Y_op=Y_op, Ts=ts, up_sample=up_sample,
                              **_unit_norm(), **kw)
    blk = rebuild_float64(blk)
    blk.cc_init = torch.as_tensor(cc, dtype=torch.float64).reshape(1, 3).clone()
    return blk.to(dtype)


def make_plain_block(Y_op=None, ts=tp.TS, up_sample=1, dtype=torch.float64):
    blk = Gantry_State_Block(Y_op=Y_op, Ts=ts, up_sample=up_sample, **_unit_norm())
    return rebuild_float64(blk).to(dtype)
