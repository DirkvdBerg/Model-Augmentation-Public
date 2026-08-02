"""The training baseline, given the truth's STATIC mass distribution at delta_a = 0.

WHAT IS CORRECTED AND WHY
-------------------------
The truth's X-Theta coupling is `B12 - mh*Y - ma*L0 - ma*delta_a` while the
baseline's is `B12 - mh*Y` (`scripts/gantry/msd-offset/plant.py`, `M8` vs the
6-state form). Setting `delta_a = 0` leaves a purely STATIC difference: a
centre-of-gravity offset `ma*L0 = 0.1010 kg*m` in `M[0,1]`, and a Theta inertia
difference `ma*(2*Y*L0 + L0^2)` in `M[1,1]`. Those terms enter the X and Theta
rows only, never Y.

This is CONFOUND REMOVAL, NOT A FIX. It has been measured
(`scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md` F4): correcting it moved
the settled X offset -4.1270e-06 -> -4.0378e-06 (2.2 % better) and Y
-3.3142e-05 -> -3.5987e-05 (8.6 % WORSE). Do not expect it to move the offset.
What it does buy is that after the correction the ONLY difference between truth
and baseline is the absorber's DYNAMICS, which is the thing the ANN is supposed
to learn.

WHY THE LFR FORM SURVIVES THE CORRECTION
----------------------------------------
`diag_attribution.py` in the coulomb-offset thread did this on the collapsed
3-DOF realization, noting the LFR block "builds M(Y) implicitly through its
polynomial constants, so the CoG term cannot be edited in one line there". It
cannot be edited in one line, but it CAN be edited exactly, because the
correction preserves the rational structure. With

    A  = alpha = m1+m2+mb+mh                (unchanged)
    B  = beta - ma*l0                       (beta = (m1-m2)*Lb/2)
    Gp = gamma + mh*d^2 + ma*l0^2           (gamma EXCLUDES mh*d^2, as in gantry_ss)

the corrected mass matrix is

    M_c(Y) = [[A,        B - mh*Y,                        0    ],
              [B - mh*Y, Gp + 2*ma*l0*Y + mh*Y^2,        -mh*d ],
              [0,        -mh*d,                           mh   ]]

which is still quadratic in Y, so adj(M_c) is still quadratic and det(M_c) is
still quadratic. Writing the cofactors out (m = mh, dd = d):

    N0 = [[m*Gp - m^2*dd^2, -m*B,      -m*dd*B     ],
          [-m*B,             A*m,       A*m*dd     ],
          [-m*dd*B,          A*m*dd,    A*Gp - B^2 ]]
    N1 = [[2*m*ma*l0,        m^2,       dd*m^2     ],
          [m^2,              0,         0          ],
          [dd*m^2,           0,         2*(A*ma*l0 + B*m)]]
    N2 = [[m^2, 0, 0], [0, 0, 0], [0, 0, A*m - m^2]]
    d(Y) = m * [ (A*Gp - A*m*dd^2 - B^2)
                 + Y*2*(A*ma*l0 + B*m)
                 + Y^2*m*(A - m) ]

At `ma = 0` every line above collapses onto `build_poly_constants` /
`d0 = mh*(alpha*gamma - beta^2)` term for term. That is gate C1, and it is why
this is a derivation rather than a re-parameterisation.

`deriv()` is restated rather than patched for the same reason `plant_coulomb.py`
restates it: `model_augmentation/` is Jan's framework and is not modified. The
arithmetic is kept TERM FOR TERM as the parent writes it (Horner, divide after
the matmul) so that the ma = 0 arm stays as close to bit-identical as floating
point allows -- reassociating costs ~2 ulp and that is exactly what broke the
equivalent gate in the Coulomb thread (trap T5).
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from model_augmentation.fit_systems.blocks import Gantry_State_Block          # noqa: E402
from model_augmentation.systems.gantry_ss import (                            # noqa: E402
    m1 as _m1, m2 as _m2, mb as _mb, mh as _mh, Jb as _Jb, Jh as _Jh,
    Lb as _Lb, d as _d, M1 as _M1, M2 as _M2, K as _K, C as _C,
    build_G_matrix_entries)

# Absorber parameters of the augmentation dataset. Single source is
# scripts/gantry/gantry_dynamic/oracle.py (ma_frac = 0.10, L0 = 0.10 m); the
# .mat files do not store them. Fitted from the data, not assumed:
# docs/msd-offset-mechanism-2026-07-29.md section 2.
from gantry_dynamic.oracle import MA as _MA, L0 as _L0                        # noqa: E402


def cog_constants(ma, l0, dtype=torch.float64):
    """(N0, N1, N2, dpoly, M0_c, M1_c, M2_c) for the CoG-corrected M(Y).

    dpoly = (c0, c1, c2) with det(M_c(Y)) = mh * (c0 + c1*Y + c2*Y^2).
    ma = 0 reproduces the uncorrected baseline exactly.
    """
    m1, m2, mb, m = (t.to(dtype) for t in (_m1, _m2, _mb, _mh))
    Jb, Jh, Lb, dd = (t.to(dtype) for t in (_Jb, _Jh, _Lb, _d))
    ma = torch.as_tensor(ma, dtype=dtype)
    l0 = torch.as_tensor(l0, dtype=dtype)

    A = m1 + m2 + mb + m                             # alpha
    beta = (m1 - m2) * Lb / 2
    gamma = Jb + Jh + (m1 + m2) * Lb ** 2 / 4        # WITHOUT mh*d^2 (gantry_ss convention)
    B = beta - ma * l0
    Gp = gamma + m * dd ** 2 + ma * l0 ** 2

    z = torch.zeros((), dtype=dtype)
    N0 = torch.stack([
        torch.stack([m * Gp - m ** 2 * dd ** 2, -m * B,      -m * dd * B]),
        torch.stack([-m * B,                     A * m,       A * m * dd]),
        torch.stack([-m * dd * B,                A * m * dd,  A * Gp - B ** 2]),
    ])
    N1 = torch.stack([
        torch.stack([2 * m * ma * l0, m ** 2,  dd * m ** 2]),
        torch.stack([m ** 2,          z,       z]),
        torch.stack([dd * m ** 2,     z,       2 * (A * ma * l0 + B * m)]),
    ])
    N2 = torch.stack([
        torch.stack([m ** 2, z, z]),
        torch.stack([z,      z, z]),
        torch.stack([z,      z, A * m - m ** 2]),
    ])
    dpoly = (A * Gp - A * m * dd ** 2 - B ** 2,
             2 * (A * ma * l0 + B * m),
             m * (A - m))

    M0_c = torch.zeros(3, 3, dtype=dtype)
    M0_c[0, 0] = A
    M0_c[0, 1] = M0_c[1, 0] = B
    M0_c[1, 1] = Gp
    M0_c[1, 2] = M0_c[2, 1] = -m * dd
    M0_c[2, 2] = m
    M1_c = _M1.to(dtype).clone()
    M1_c[1, 1] = 2 * ma * l0
    M2_c = _M2.to(dtype).clone()
    return N0, N1, N2, dpoly, M0_c, M1_c, M2_c


def mass_matrix_cog(Y, ma=float(_MA), l0=float(_L0)):
    """The corrected 3x3 M(Y) as plain numpy, for the independent gate."""
    m1, m2, mb, m = float(_m1), float(_m2), float(_mb), float(_mh)
    Jb, Jh, Lb, dd = float(_Jb), float(_Jh), float(_Lb), float(_d)
    A = m1 + m2 + mb + m
    beta = (m1 - m2) * Lb / 2
    gamma = Jb + Jh + (m1 + m2) * Lb ** 2 / 4
    off = beta - m * Y - ma * l0
    # ma*l0**2 is part of the STATIC Theta inertia; dropping it was caught by gate
    # C3 as a flat 1.010e-02 kg*m^2 = ma*L0^2 offset. Kept as a written-out
    # expression rather than reusing Gp, so this stays an INDEPENDENT check of
    # cog_constants rather than a restatement of it.
    g22 = gamma + m * dd ** 2 + ma * l0 ** 2 + 2 * ma * l0 * Y + m * Y ** 2
    return np.array([[A, off, 0.0], [off, g22, -m * dd], [0.0, -m * dd, m]])


class Gantry_State_Block_CoG(Gantry_State_Block):
    """Gantry_State_Block whose M(Y) carries the absorber's static mass at da = 0.

    Parameters
    ----------
    ma, l0 : float
        Absorber mass [kg] and equilibrium offset along +Y [m]. `ma = 0` is the
        uncorrected baseline and is the no-op gate.
    """

    def __init__(self, ma=float(_MA), l0=float(_L0), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ma_cog, self.l0_cog = float(ma), float(l0)
        N0, N1, N2, dpoly, _M0c, M1c, M2c = cog_constants(ma, l0, dtype=torch.float64)
        d0 = _mh.to(torch.float64) * dpoly[0]                 # det(M_c(0))
        Ax, Bw, Bu, A_comb = build_G_matrix_entries(
            N0, d0, M1c, M2c, _K.to(torch.float64), _C.to(torch.float64))

        # Overwrite the parent's buffers. The parent keeps mh/alpha/beta/gamma_
        # ONLY to form d(Y); this subclass forms d(Y) from dpoly instead, so
        # those four are left untouched and unused.
        # Constants are DERIVED and STORED in float64 and the whole block is then
        # promoted, so that a later .to(float64) recovers full precision rather
        # than an upcast float32 value. Callers cast down if they want float32.
        self.N0, self.N1, self.N2 = N0, N1, N2
        self.Ax, self.Bw, self.Bu, self.A_combined = Ax, Bw, Bu, A_comb
        self.register_buffer('dpoly', torch.stack(list(dpoly)))
        if self.Y_op is not None:
            Y_t = torch.tensor(self.Y_op, dtype=torch.float64)
            self.N_op = N0 + N1 * Y_t + N2 * Y_t ** 2
            self.d_op = (_mh.to(torch.float64)
                         * (dpoly[0] + dpoly[1] * Y_t + dpoly[2] * Y_t ** 2))
        self.to(torch.float64)

    def deriv(self, x, u):
        # Restated from Gantry_State_Block.deriv (blocks.py:781-834). The ONLY
        # change is d(Y): the parent hard-codes it from (alpha, beta, gamma_),
        # which no longer describes the corrected M(Y). Everything else is the
        # parent's arithmetic, term for term.
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = self._mats()
        c0, c1, c2 = self.dpoly[0], self.dpoly[1], self.dpoly[2]

        x_phys = x * self.std_x + self.x_mean
        u_phys = u * self.std_u + self.u_mean
        x2 = x_phys.squeeze(-1)
        u2 = u_phys.squeeze(-1)
        u_log = u2 @ self.P_mat.T

        fnet = (-(x2[:, :3] @ K_mat.T)
                - (x2[:, 3:] @ C_mat.T)
                + u_log)

        if self.Y_op is not None:
            Y_val = self.Y_op
            a = (self.N_op @ fnet.T).T / self.d_op
        else:
            Y = x2[:, 2]
            dY = mh * (c0 + Y * (c1 + Y * c2))               # Horner, as the parent does
            Y_r = Y.unsqueeze(0)
            n0f = N0 @ fnet.T
            n1f = N1 @ fnet.T
            n2f = N2 @ fnet.T
            a = (n0f + Y_r * (n1f + Y_r * n2f)).T / dY[:, None]
            Y_val = Y[:, None]

        z = torch.cat([a, Y_val * a], dim=-1)
        w = Y_val * z
        combined = torch.cat([x2, w, u_log], dim=-1)
        xdot_phys = combined @ A_combined.T
        return (xdot_phys / self.std_x_1d).unsqueeze(-1)


def make_block(Y_op=None, cog=True, ts=1 / 4000, up_sample=1,
               dtype=torch.float64, std_x=None, std_u=None,
               x_mean=None, u_mean=None):
    """Unnormalised (std = 1, mean = 0) block in straight physical units.

    Normalisation is a training concern; for a replay it only adds a way to be
    wrong (`plant_coulomb.make_block` takes the same position).
    """
    kw = dict(
        Y_op=Y_op, Ts=ts, up_sample=up_sample,
        std_x=np.ones((6, 1)) if std_x is None else std_x,
        std_u=np.ones((3, 1)) if std_u is None else std_u,
        x_mean=np.zeros((6, 1)) if x_mean is None else x_mean,
        u_mean=np.zeros((3, 1)) if u_mean is None else u_mean,
    )
    blk = (Gantry_State_Block_CoG(ma=float(_MA), l0=float(_L0), **kw) if cog
           else Gantry_State_Block(**kw))
    return blk.to(dtype)


def rollout_batch(block, x0, u_stage, n_out=6):
    """Batched open-loop free run. x0 (B, 6), u_stage (B, N, 3) PHYSICAL stage forces.

    Returns (B, N, n_out) physical states. Batching is what makes the per-window
    check cheap: every window is one row and they all step together.
    """
    dtype = block.P_mat.dtype
    B, N, _ = u_stage.shape
    out = np.empty((B, N, n_out))
    x = torch.as_tensor(np.asarray(x0, float).reshape(B, 6, 1), dtype=dtype)
    u_t = torch.as_tensor(np.asarray(u_stage, float), dtype=dtype)
    with torch.no_grad():
        for k in range(N):
            out[:, k] = x[:, :n_out, 0].numpy()
            x = block.nonlinear_function(torch.cat([x, u_t[:, k].reshape(B, 3, 1)], dim=1))
    return out
