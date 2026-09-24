"""Baseline LPV-LFR gantry block with Garcia dry friction, plus the record loader.

The BASELINE here is the real one: `Gantry_State_Block` from
`model_augmentation/fit_systems/blocks.py`, the LFR rational form that
`scripts/gantry/gantry_interconnect_dynamic.py` actually trains against. It is NOT
the collapsed 3-DOF `deriv6` in `scripts/gantry/msd-offset/plant.py`; that is a
different realization of the same physics and an offset measured against it would
be an offset in a lookalike rather than in the baseline.

`model_augmentation/` is Jan's framework and is NOT modified. Friction is added by
SUBCLASSING and overriding `deriv()`, which is why the whole method is restated
here rather than patched.

THE ONE WAY TO GET THIS WRONG
-----------------------------
`u_log` enters `Gantry_State_Block.deriv` at TWO sites: the net force

    blocks.py:799-801
        fnet = (-(x2[:, :3] @ K_mat.T)
                - (x2[:, 3:] @ C_mat.T)
                + u_log)                 # (batch, 3)

and again directly through G,

    blocks.py:829-830
        combined  = torch.cat([x2, w, u_log], dim=-1)   # (batch, 15)
        xdot_phys = combined @ A_combined.T              # (batch, 6)

At Y = 0 we have w = 0, so the acceleration comes ONLY from the direct Bu*u_log
term. Subtracting friction inside `fnet` alone makes it silently vanish at Y = 0
and weakens it everywhere else. This is invariant 2 of
`Matlab-scripts/coulomb-friction/PLAN.md`; on the sibling LPV-LFR implementation
an fnet-only insertion was measured to diverge from the direct collapsed form by
1.17 m/s^2 at Y = 0. `fnet_only=True` reproduces that trap on purpose, as a
negative control; it is never a production setting.

G IS UNCHANGED. G is built from N0, d0, M1, M2, K, C and `u` never enters it, so
`cc` never touches it. Friction rides the existing M(Y)^-1 loop. (PLAN.md
invariant 3.)

FRAME. Friction acts per PHYSICAL rail, so it is built in STAGE coordinates and
projected back with the block's own P:

    v_stage = P.T @ qdot_logical        Fc_log = P @ (cc * phi(v_stage))

Note the block's input convention differs from the MATLAB EOM: here `u` is in
STAGE forces and `u_log = P @ u`, whereas `gantrySystemExtendedCoulomb.m` takes
`u` already logical. The friction term itself is identical in both.

NORMALISATION. The block carries normalised x and u. Friction must be built from
the PHYSICAL velocity, i.e. after the denormalisation at the top of deriv(), never
from the normalised state.

sign VS tanh
------------
Default is hard `sign`, matching Garcia 2013 Fig. 2 and matching the MATLAB truth
in `Matlab-scripts/Augmentation-coulomb/`. For the offset replay that is required,
not preferred: the replay difference is model mismatch ONLY if truth and baseline
carry the identical friction law. D-116's `tanh` argument is about
differentiability and becomes live again only if friction enters a TRAINING run,
which is why the switch exists rather than being hard-coded (D-138).
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch
from scipy.io import loadmat

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)

from model_augmentation.fit_systems.blocks import Gantry_State_Block
from model_augmentation.systems.gantry_ss import (
    P as _P_t, Lb as _Lb_t, m1 as _M1_, m2 as _M2_, mb as _MB_, mh as _MH_)

P_np = _P_t.numpy().astype(np.float64)
LB = float(_Lb_t)

# THEORY: garcia2013 (garcia2013_gantry-decoupling-control.pdf, Fig. 2 and the
# identified-parameter table) -- per-actuator Coulomb friction of the H-gantry,
# identified by displacing each axis at constant velocity. Same values as
# kamtin-fp-model/03 Simulink gantry/main.m and gtd_config.m. NOT a knob: there is
# no sweep and nothing here is tuned to make the offset behave (D-138).
CC_GARCIA = (16.80, 18.35, 11.60)     # (cc1, cc2, ccy) [N], STAGE coordinates

# HEURISTIC: tanh transition width, used ONLY when mode='tanh'. Carried from
# D-116 / coulomb_lfr.py. Not derived. Irrelevant in the default sign mode.
V0_DEFAULT = 1e-3                     # [m/s]

TRAJ_COULOMB = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory',
                            'augmentation_coulomb')
TRAJ_PLAIN = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory',
                          'augmentation')


class Gantry_State_Block_Coulomb(Gantry_State_Block):
    """Gantry_State_Block + Garcia dry friction as an effective input.

    Parameters
    ----------
    cc : tuple(float, float, float)
        (cc1, cc2, ccy) Coulomb magnitudes in STAGE coordinates [N]. All-zero
        must reproduce the parent bit-for-bit; that is gate G1.
    mode : {'sign', 'tanh'}
        Friction law. 'sign' is Garcia's and is required for the offset replay.
        'tanh' is the D-116 differentiable surrogate, for training only.
    v0 : float
        tanh transition width [m/s]. Ignored when mode='sign'.
    fnet_only : bool
        NEGATIVE CONTROL ONLY. Subtract friction at the fnet site but not at the
        direct G site, reproducing the documented Y = 0 trap. Never use for a
        production run.
    """

    def __init__(self, cc=CC_GARCIA, mode='karnopp', v0=V0_DEFAULT,
                 fnet_only=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ('karnopp', 'sign', 'tanh'):
            raise ValueError(
                f"mode must be 'karnopp', 'sign' or 'tanh', got {mode!r}")
        self.mode = mode
        self.v0 = float(v0)
        self.fnet_only = bool(fnet_only)
        self.register_buffer(
            'cc', torch.as_tensor(cc, dtype=self.P_mat.dtype).reshape(1, 3))
        # HEURISTIC: V_EPS = (cc1+cc2)/m_total * ts. IDENTICAL expression to the
        # MATLAB EOM (gantrySystemExtendedCoulomb.m); it must stay identical or
        # the replay difference stops being a pure model mismatch. m_total is the
        # X-row mass, matching the MATLAB m1+m2+mb+mh+ma (mh there is mh_rigid).
        m_total = float(_M1_ + _M2_ + _MB_ + _MH_)
        self._v_eps = float((cc[0] + cc[1]) / m_total * self.Ts)

    def _phi(self, v_stage):
        """Smoothed or hard sign of the STAGE-frame velocity."""
        if self.mode == 'tanh':
            return torch.tanh(v_stage / self.v0)
        # THEORY: garcia2013 Fig. 2 -- F_c,i = cc_i * sign(v_i), per actuator
        return torch.sign(v_stage)

    def friction_logical(self, qdot_log):
        """Slip-branch F_c in LOGICAL coordinates. Physical units.

        This is the SLIP force only. In 'karnopp' mode it is what a rail carries
        while sliding; the stuck rails are solved for in deriv(). Kept as a
        method because the gates use it to check the frame and the law directly.
        """
        # Row-vector convention: (P.T @ q) == q @ P, and (P @ F) == F @ P.T
        v_stage = qdot_log @ self.P_mat            # [dX1, dX2, dY]  STAGE
        Fc_stage = self.cc * self._phi(v_stage)    # (batch, 3)      STAGE  [N]
        return Fc_stage @ self.P_mat.T             # (batch, 3)      LOGICAL

    def _stage_friction_karnopp(self, x2, u_log, a_free, M_r, Y_val):
        """Set-valued Coulomb friction in STAGE coordinates (Karnopp).

        A stuck rail's force is whatever holds its acceleration at zero, so it is
        SOLVED, not evaluated. Friction enters the LFR derivative AFFINELY at both
        u sites, so the acceleration is

            a = a_free - Dm @ F_stage

        and the map Dm is derived analytically rather than by finite difference:

            a      = Ax_v x + Bw_v w + Bu_v u_eff        (velocity rows of G)
            a_lfr  = M_r fnet,   fnet = -Kq - Cqd + u_eff   =>  d a_lfr/d u_eff = M_r
            w      = [Y a_lfr ; Y^2 a_lfr]              =>  d w/d u_eff = [Y M_r ; Y^2 M_r]
            d a/d u_eff = Bw_v [Y M_r ; Y^2 M_r] + Bu_v
            Dm     = (d a/d u_eff) @ P,      G = P.T @ Dm

        Note this picks up BOTH u sites automatically: Bu_v is the direct term and
        the Bw_v part is the route through the LFR loop. Deriving it from only one
        would reproduce the Y = 0 trap in the friction solve itself.
        """
        A = self.A_combined
        Bw_v = A[3:6, 6:12]                       # (3,6)
        Bu_v = A[3:6, 12:15]                      # (3,3)
        Yv = Y_val if torch.is_tensor(Y_val) else torch.as_tensor(
            Y_val, dtype=x2.dtype, device=x2.device)
        Yv = Yv.reshape(-1, 1, 1).expand(x2.shape[0], 1, 1)

        Mstack = torch.cat([Yv * M_r, Yv ** 2 * M_r], dim=-2)     # (B,6,3)
        dadu = Bw_v @ Mstack + Bu_v                                # (B,3,3)
        Dm = dadu @ self.P_mat                                     # (B,3,3)
        G = self.P_mat.T @ Dm                                      # (B,3,3)

        v_stage = x2[:, 3:] @ self.P_mat                           # (B,3)
        a_free_stage = a_free @ self.P_mat                         # (B,3)
        cc = self.cc                                               # (1,3)

        # HEURISTIC: V_EPS = (cc1+cc2)/m_total * ts, the velocity dry friction
        # removes in ONE step. Identical expression to the MATLAB EOM, and it must
        # stay identical: a different band is a different model and the replay
        # difference would stop being a model mismatch.
        v_eps = self._v_eps

        stuck = v_stage.abs() < v_eps                              # (B,3) bool
        Fassign = cc * torch.sign(v_stage)                         # slip forces
        eye = torch.eye(3, dtype=x2.dtype, device=x2.device).expand(x2.shape[0], 3, 3)

        F = torch.zeros_like(v_stage)
        for _ in range(4):          # active set: 3 rails, converges fast
            m = stuck.unsqueeze(-1).to(x2.dtype)                   # (B,3,1)
            Amat = m * G + (1 - m) * eye
            bvec = (stuck.to(x2.dtype) * a_free_stage
                    + (~stuck).to(x2.dtype) * Fassign).unsqueeze(-1)
            F = torch.linalg.solve(Amat, bvec).squeeze(-1)         # (B,3)
            # A stuck rail needing more than cc breaks away, saturating at cc in
            # the direction of the force it could NOT hold. Using sign(v) here
            # would be wrong: a stuck rail has |v| < V_EPS so that sign is
            # arbitrary, and sign(0) = 0 would remove the friction entirely.
            broke = stuck & (F.abs() > cc)
            if not bool(broke.any()):
                break
            Fassign = torch.where(broke, cc * torch.sign(F), Fassign)
            stuck = stuck & ~broke
        return F

    def deriv(self, x, u):
        # Restated from Gantry_State_Block.deriv (blocks.py:781-834). The ONLY
        # changes are the friction term and u_log -> u_eff at BOTH u sites.
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = self._mats()

        # --- denormalise -------------------------------------------------
        x_phys = x * self.std_x + self.x_mean
        u_phys = u * self.std_u + self.u_mean

        x2 = x_phys.squeeze(-1)          # (batch, 6)  PHYSICAL
        u2 = u_phys.squeeze(-1)          # (batch, 3)  PHYSICAL, STAGE forces

        # --- P transform: stage forces -> logical forces -----------------
        u_log = u2 @ self.P_mat.T        # (batch, 3)

        # --- Y-dependent quantities --------------------------------------
        if self.Y_op is not None:
            Y_val = self.Y_op
            Y_sc = torch.full((x2.shape[0],), float(self.Y_op),
                              dtype=x2.dtype, device=x2.device)
        else:
            Y = x2[:, 2]
            dY = mh * (alpha * gamma_ - beta ** 2
                       + 2 * beta * mh * Y
                       + mh * (alpha - mh) * Y ** 2)
            Y_r = Y.unsqueeze(0)
            Y_val = Y[:, None]
            Y_sc = Y

        def _forward(u_eff, u_direct):
            """One LFR pass. u_eff feeds fnet (site 1), u_direct feeds G (site 2).

            The arithmetic here is kept TERM FOR TERM as Gantry_State_Block.deriv
            writes it: the frozen branch divides by d_op AFTER the matmul and the
            LPV branch uses Horner. Precomputing N/d and multiplying instead is
            algebraically identical but reassociates the floating point, which
            costs ~2 ulp and broke the cc = 0 bit-identity gate. That gate is
            exact by design, so the order stays.
            """
            fnet = (-(x2[:, :3] @ K_mat.T)
                    - (x2[:, 3:] @ C_mat.T)
                    + u_eff)                                   # <- site 1
            if self.Y_op is not None:
                a = (self.N_op @ fnet.T).T / self.d_op
            else:
                n0f = N0 @ fnet.T
                n1f = N1 @ fnet.T
                n2f = N2 @ fnet.T
                a = (n0f + Y_r * (n1f + Y_r * n2f)).T / dY[:, None]
            z = torch.cat([a, Y_val * a], dim=-1)
            w = Y_val * z
            # SITE 2. At Y = 0, w = 0, so this is the ONLY path to acceleration.
            combined = torch.cat([x2, w, u_direct], dim=-1)
            return combined @ A_combined.T

        def _M_rational():
            """M(Y)^-1 as an explicit 3x3, for the friction Jacobian only.

            Used solely to build G in the stuck-force solve, where a 2 ulp
            difference is irrelevant; it is never on the no-op path.
            """
            if self.Y_op is not None:
                return (self.N_op / self.d_op).expand(x2.shape[0], 3, 3)
            return ((N0 + Y[:, None, None] * N1 + (Y ** 2)[:, None, None] * N2)
                    / dY[:, None, None])

        # --- Coulomb friction as an effective input (THE addition) -------
        if self.mode == 'karnopp':
            # A stuck rail's force is solved, not evaluated, so the frictionless
            # acceleration is needed first. One extra LFR pass, no finite
            # differences: the map from friction to acceleration is analytic.
            a_free = _forward(u_log, u_log)[:, 3:]
            F_stage = self._stage_friction_karnopp(
                x2, u_log, a_free, _M_rational(), Y_sc)
            Fc_log = F_stage @ self.P_mat.T
        else:
            Fc_log = self.friction_logical(x2[:, 3:])

        u_eff = u_log - Fc_log
        # fnet_only=True deliberately leaves u_log at site 2 to expose the Y = 0
        # trap. Never a production setting.
        xdot_phys = _forward(u_eff, u_log if self.fnet_only else u_eff)

        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot


def make_block(Y_op, cc=CC_GARCIA, mode='sign', ts=1 / 4000, up_sample=1,
               fnet_only=False, dtype=torch.float64):
    """Unnormalised block (std = 1, mean = 0), i.e. straight physical units.

    Normalisation is a training concern; for a replay it only adds a way to be
    wrong, so it is set to identity here and the block is fed physical values.
    """
    blk = Gantry_State_Block_Coulomb(
        cc=cc, mode=mode, fnet_only=fnet_only,
        Y_op=Y_op, Ts=ts, up_sample=up_sample,
        std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1)),
    )
    return blk.to(dtype)


def rollout(block, x0, u_stage, n_out=3):
    """Open-loop free run. u_stage is (N, 3) PHYSICAL stage forces, held per step.

    Returns the first n_out states per sample, physical units.
    """
    dtype = block.P_mat.dtype
    N = len(u_stage)
    out = np.empty((N, n_out))
    x = torch.as_tensor(np.asarray(x0, float).reshape(1, 6, 1), dtype=dtype)
    u_t = torch.as_tensor(np.asarray(u_stage, float), dtype=dtype)
    with torch.no_grad():
        for k in range(N):
            out[k] = x[0, :n_out, 0].numpy()
            uk = u_t[k].reshape(1, 3, 1)
            x = block.nonlinear_function(torch.cat([x, uk], dim=1))
    return out


def load_record(name, traj_dir, fs_new=4000):
    """Load one record at fs_new. u block-mean (D-087), y/states point-sampled.

    Mirrors msd-offset/plant.py::load_record, but the trajectory directory is an
    ARGUMENT rather than a module constant, because this investigation needs the
    augmentation_coulomb dataset and that one is hard-wired to augmentation.
    """
    dm = loadmat(os.path.join(traj_dir, name + '.mat'), squeeze_me=True)
    ts0 = float(dm['dt'])
    D = int(round((1.0 / ts0) / fs_new))
    u0 = np.asarray(dm['u_total'], float)
    n = len(u0) // D
    rec = dict(
        name=name, ts=ts0 * D, D=D, fs=fs_new, traj_dir=traj_dir,
        u=u0[:n * D].reshape(n, D, 3).mean(axis=1),          # block mean (D-087)
        y=np.asarray(dm['y'], float)[::D][:n],
        x_logical=np.asarray(dm['x_logical'], float)[::D][:n],
        r=np.asarray(dm['r_sim'], float)[::D][:n],
    )
    for opt in ('delta_a', 'vdelta_a'):
        if opt in dm:
            rec[opt] = np.asarray(dm[opt], float)[::D][:n]
    rec['u_log'] = (P_np @ rec['u'].T).T                      # stage -> logical
    rec['Y_op'] = float(rec['y'][0, 2])
    rec['t'] = np.arange(n) * rec['ts']
    return rec


def true_ic(rec, k):
    """Baseline 6-state initial condition from the record's own state at sample k."""
    return rec['x_logical'][k].copy()
