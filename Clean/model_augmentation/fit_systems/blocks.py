from typing import Callable, Type, Union

import numpy as np
import torch
from torch import Tensor, nn

from model_augmentation.utils.utils import to_tensor, added


class Block(nn.Module):
    # CHANGED (clean): trimmed to the spec path
    """Basic block implementation with variables that need to be defined for interconnect."""

    def __init__(self, nz, nw, name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.nw = nw
        self.nz = nz

        self.name = name
        self.block_ix = None

    def init_block(self, z: Tensor):
        return


class Static_ANN_Block(Block):
    def __init__(
        self,
        net: Union[
            Type[nn.Module], Callable[..., nn.Module]
        ],
        n_nodes_per_layer=64,
        n_hidden_layers=2,
        activation=nn.Tanh,
        net_kwargs=None,
        *args,
        **kwargs,
    ) -> None:
        # CHANGED (clean): trimmed to the spec path
        super(Static_ANN_Block, self).__init__(*args, **kwargs)

        # CHANGED: allow a network to receive its structural initialization data at normal
        # construction time. Existing callers pass nothing and retain the exact old call.
        self.net = net(
            n_in=self.nz,
            n_out=self.nw,
            n_nodes_per_layer=n_nodes_per_layer,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            **({} if net_kwargs is None else net_kwargs),
        )
        # CHANGED: per-output intervention gate for the trajectory-orthogonality method.
        self.register_buffer('out_gate', torch.ones(self.nw), persistent=False)

    def forward(self, z: Tensor):
        # CHANGED (clean): trimmed to the spec path
        w = self.net(z.view(-1, self.nz))
        w = w * self.out_gate
        return w.view(-1, self.nw, 1)


@added
class Linear_Output_Block(Block):
    def __init__(
        self, C=torch.empty((0, 0)), D=torch.empty((0, 0)), *args, **kwargs
    ) -> None:
        _C = to_tensor(C)
        _D = to_tensor(D)

        self.ny = _C.size(0) if _C.numel() else 0
        self.nx = _C.size(1) if _C.numel() else 0
        self.nu = _D.size(1) if _D.numel() else 0

        super().__init__(nw=self.ny, nz=self.nx + self.nu, *args, **kwargs)

        self.register_buffer("C", _C)
        self.register_buffer("D", _D)

    def forward(self, z: Tensor):
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w


class Discrete_Nonlinear_Function_Block(Block):
    # CHANGED (clean): trimmed to the spec path
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, z: Tensor, mats=None):
        # CHANGED (clean): trimmed to the spec path
        # CHANGED (D-195): optional per-pass constants, forwarded to `nonlinear_function` only
        # when supplied.
        w = self.nonlinear_function(z) if mats is None else self.nonlinear_function(z, mats)
        return w


@added
class Gantry_State_Block(Discrete_Nonlinear_Function_Block):
    """
    Gantry continuous-time ODE integrated with RK4, using the LFR rational structure."""

    def __init__(
        self,
        Y_op: float = 0.3,
        std_x=np.ones((6, 1)),
        std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)),
        u_mean=np.zeros((3, 1)),
        Ts: float = 1 / 20000,
        up_sample: int = 10,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(nz=9, nw=6, *args, **kwargs)

        self.Ts   = Ts
        self.up_sample = up_sample
        self.nu   = 3
        self.ny   = 3
        self.nx   = 6
        self.Y_op = Y_op

        from model_augmentation.systems.gantry_ss import (
            mh as _mh, m1 as _m1, m2 as _m2, mb as _mb,
            Jb as _Jb, Jh as _Jh, Lb as _Lb, d as _d,
            M1 as _M1, M2 as _M2, K as _K, C as _C, P as _P,
            build_poly_constants, build_G_matrix_entries,
        )

        _alpha, _beta, _gamma, _N0, _N1, _N2 = build_poly_constants(
            _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
        )
        _d0 = _mh * (_alpha * _gamma - _beta ** 2)

        _Ax, _Bw, _Bu, _A_combined = build_G_matrix_entries(_N0, _d0, _M1, _M2, _K, _C)

        self.register_buffer("mh",     _mh.clone())
        self.register_buffer("alpha",  _alpha.clone())
        self.register_buffer("beta",   _beta.clone())
        self.register_buffer("gamma_", _gamma.clone())
        self.register_buffer("N0",     _N0.clone())
        self.register_buffer("N1",     _N1.clone())
        self.register_buffer("N2",     _N2.clone())

        self.register_buffer("Ax",         _Ax.clone())
        self.register_buffer("Bw",         _Bw.clone())
        self.register_buffer("Bu",         _Bu.clone())
        self.register_buffer("A_combined", _A_combined.clone())

        self.register_buffer("K_mat", _K.clone())
        self.register_buffer("C_mat", _C.clone())
        self.register_buffer("P_mat", _P.clone())

        self.register_buffer("std_x",    to_tensor(std_x).reshape(6, 1))
        self.register_buffer("std_x_1d", to_tensor(std_x).reshape(6))
        self.register_buffer("std_u",    to_tensor(std_u).reshape(3, 1))
        self.register_buffer("x_mean",   to_tensor(x_mean).reshape(6, 1))
        self.register_buffer("u_mean",   to_tensor(u_mean).reshape(3, 1))

    def nonlinear_function(self, z: Tensor):
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        # CHANGED (D-192): the RK4 loop and the derivative body take the parameter-dependent
        # matrix tuple EXPLICITLY (`_rk4`, `_deriv_with`) so the one-step OBC path can integrate
        # the same transition at a caller-supplied parameter point without writing `_cur`.
        return self._rk4(x, u, self._mats())

    def _rk4(self, x: Tensor, u: Tensor, mats) -> Tensor:
        for i in range(self.up_sample):
            k1 = (self.Ts / self.up_sample) * self._deriv_with(x, u, mats)
            k2 = (self.Ts / self.up_sample) * self._deriv_with(x + k1 / 2, u, mats)
            k3 = (self.Ts / self.up_sample) * self._deriv_with(x + k2 / 2, u, mats)
            k4 = (self.Ts / self.up_sample) * self._deriv_with(x + k3, u, mats)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def _deriv_with(self, x: Tensor, u: Tensor, mats, terms: dict = None) -> Tensor:
        # CHANGED (D-192): body of `deriv` with the matrix tuple passed in; see nonlinear_function.
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats

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
            Y   = x2[:, 2]
            dY  = mh * (alpha * gamma_ - beta ** 2
                        + 2 * beta * mh * Y
                        + mh * (alpha - mh) * Y ** 2)
            Y_r = Y.unsqueeze(0)
            n0f = N0 @ fnet.T
            n1f = N1 @ fnet.T
            n2f = N2 @ fnet.T
            a   = (n0f + Y_r * (n1f + Y_r * n2f)).T / dY[:, None]
            Y_val = Y[:, None]

        z = torch.cat([a, Y_val * a], dim=-1)
        w = Y_val * z

        combined  = torch.cat([x2, w, u_log], dim=-1)
        xdot_phys = combined @ A_combined.T

        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot

    # CHANGED (D-194): primal AND tangent in ONE pass.
    def _deriv_both_with(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats,
                         terms: dict = None, sterms: dict = None):
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats
        (dK, dC, dA_combined,
         dmh, dalpha, dbeta, dgamma_, dN0, dN1, dN2) = dmats

        if self.Y_op is not None:
            raise NotImplementedError(
                'the hand-written tangent covers the LPV branch (Y_op=None) only. The frozen-Y '
                'branch uses the precomputed N_op/d_op buffers, whose tangent is not derived '
                'here; returning an unvalidated tangent silently would be worse than refusing. '
                'The production configuration is Y_op=None (D-192).')

        s_x2 = (sx * self.std_x).squeeze(-1)
        x2 = (x * self.std_x + self.x_mean).squeeze(-1)
        u2 = (u * self.std_u + self.u_mean).squeeze(-1)
        u_log = u2 @ self.P_mat.T

        fnet = -(x2[:, :3] @ K_mat.T) - (x2[:, 3:] @ C_mat.T) + u_log
        s_fnet = (-(s_x2[:, :3] @ K_mat.T) - (x2[:, :3] @ dK.T)
                  - (s_x2[:, 3:] @ C_mat.T) - (x2[:, 3:] @ dC.T))

        Y, sY = x2[:, 2], s_x2[:, 2]
        poly = (alpha * gamma_ - beta ** 2
                + 2 * beta * mh * Y
                + mh * (alpha - mh) * Y ** 2)
        s_poly = (dalpha * gamma_ + alpha * dgamma_ - 2 * beta * dbeta
                  + 2 * (dbeta * mh * Y + beta * dmh * Y + beta * mh * sY)
                  + (dmh * (alpha - mh) + mh * (dalpha - dmh)) * Y ** 2
                  + mh * (alpha - mh) * 2 * Y * sY)
        dY_den = mh * poly
        s_dY_den = dmh * poly + mh * s_poly

        Y_r, sY_r = Y.unsqueeze(0), sY.unsqueeze(0)
        n0f, n1f, n2f = N0 @ fnet.T, N1 @ fnet.T, N2 @ fnet.T
        s_n0f = dN0 @ fnet.T + N0 @ s_fnet.T
        s_n1f = dN1 @ fnet.T + N1 @ s_fnet.T
        s_n2f = dN2 @ fnet.T + N2 @ s_fnet.T
        num = n0f + Y_r * (n1f + Y_r * n2f)
        s_num = (s_n0f + sY_r * (n1f + Y_r * n2f)
                 + Y_r * (s_n1f + sY_r * n2f + Y_r * s_n2f))

        a = num.T / dY_den[:, None]
        s_a = (s_num.T - a * s_dY_den[:, None]) / dY_den[:, None]

        Y_val, sY_val = Y[:, None], sY[:, None]
        z = torch.cat([a, Y_val * a], dim=-1)
        s_z = torch.cat([s_a, sY_val * a + Y_val * s_a], dim=-1)
        w = Y_val * z
        s_w = sY_val * z + Y_val * s_z

        combined = torch.cat([x2, w, u_log], dim=-1)
        s_combined = torch.cat([s_x2, s_w, torch.zeros_like(u_log)], dim=-1)
        xdot_phys = combined @ A_combined.T
        s_xdot_phys = s_combined @ A_combined.T + combined @ dA_combined.T

        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        s_xdot = (s_xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot, s_xdot

    # CHANGED (D-194): RK4 carrying the sensitivity alongside the state.
    def _rk4_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        h = self.Ts / self.up_sample
        for _ in range(self.up_sample):
            d1, e1 = self._deriv_both_with(x, sx, u, mats, dmats)
            k1, s1 = h * d1, h * e1
            d2, e2 = self._deriv_both_with(x + k1 / 2, sx + s1 / 2, u, mats, dmats)
            k2, s2 = h * d2, h * e2
            d3, e3 = self._deriv_both_with(x + k2 / 2, sx + s2 / 2, u, mats, dmats)
            k3, s3 = h * d3, h * e3
            d4, e4 = self._deriv_both_with(x + k3, sx + s3, u, mats, dmats)
            k4, s4 = h * d4, h * e4
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            sx = sx + (s1 + 2 * s2 + 2 * s3 + s4) / 6
        return x, sx


@added
class _Trainable_Gantry_State_Block(Gantry_State_Block):
    """
    Shared trainable gantry transition mechanics (D-191)."""

    PARAM_NAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
                   "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]

    def __init__(self, RMSE_baseline: float = 1.0, flag_loss_reg: bool = True,
                 params_init=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        from model_augmentation.systems.gantry_ss import (
            Lb as _Lb,
            build_poly_constants as _build_poly,
            build_G_matrix_entries as _build_G,
        )
        self._build_poly = _build_poly
        self._build_G = _build_G

        params_init = to_tensor(params_init).reshape(14).clone()

        self.register_buffer("params_init", params_init)
        # CHANGED (D-191): subclasses build the regularizer in their own coordinates.
        self._RMSE_baseline = float(RMSE_baseline)
        self.flag_loss_reg = flag_loss_reg

        self.register_buffer("Lb", _Lb.clone())

        self._cur = None

    def _build_KC(self, p: Tensor):
        """Differentiable K(theta), C(theta): same structure as gantry_ss K, C.
        Uses only the identifiable sums kb1+kb2, cb1+cb2 (matrices never depend
        on the individual splits)."""
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         mh, m1, m2, mb, Jb, Jh, d) = p
        z = p.new_zeros(())
        Lb = self.Lb
        kb_sum, cb_sum = kb1 + kb2, cb1 + cb2
        K = torch.stack([
            torch.stack([z, z,      z]),
            torch.stack([z, kb_sum, z]),
            torch.stack([z, z,      z]),
        ])
        C = torch.stack([
            torch.stack([cg1 + cg2,            (cg1 - cg2) * Lb / 2,                z]),
            torch.stack([(cg1 - cg2) * Lb / 2, cb_sum + (cg1 + cg2) * Lb ** 2 / 4,  z]),
            torch.stack([z,                    z,                                   cy]),
        ])
        return K, C

    @staticmethod
    def _build_M1M2(mh: Tensor):
        """M1, M2 coefficients of M(Y): depend on mh only (gantry_ss M1, M2)."""
        z = mh.new_zeros(())
        M1 = torch.stack([
            torch.stack([z,   -mh,  z]),
            torch.stack([-mh,  z,   z]),
            torch.stack([z,    z,   z]),
        ])
        M2 = torch.stack([
            torch.stack([z,   z,   z]),
            torch.stack([z,   mh,  z]),
            torch.stack([z,   z,   z]),
        ])
        return M1, M2

    def _mats_from_raw(self, p: Tensor):
        # CHANGED (D-192): the per-timestep matrix rebuild, factored out of nonlinear_function so
        # the OBC pure transition can call it with a caller-supplied raw vector. Same operations.
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         mh, m1, m2, mb, Jb, Jh, d) = p
        alpha, beta, gamma_, N0, N1, N2 = self._build_poly(
            m1, m2, mb, mh, Jb, Jh, self.Lb, d)
        d0 = mh * (alpha * gamma_ - beta ** 2)
        M1, M2 = self._build_M1M2(mh)
        K, C = self._build_KC(p)
        _, _, _, A_combined = self._build_G(N0, d0, M1, M2, K, C)
        return (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2)

    # CHANGED (D-195): the M(Y) rebuild is a PER-PASS constant, not a per-timestep one.
    def nonlinear_function(self, z: Tensor, mats=None):
        self._cur = self._mats_from_raw(self._recover_params()) if mats is None else mats
        return super().nonlinear_function(z)

    def pass_mats(self):
        """The M(Y) structure at the CURRENT parameters, to be built once per objective."""
        mats = self._mats_from_raw(self._recover_params())
        return mats

    def _mats(self):
        return self._cur

    # CHANGED: `_cur` is a PER-FORWARD CACHE and must never be serialised.
    def __getstate__(self):
        state = dict(self.__dict__)
        state['_cur'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cur = None


@added
class Reduced_Gantry_State_Block(_Trainable_Gantry_State_Block):
    """Gantry physical block parameterized by the TEN identifiable combinations (D-190)."""

    COMBO_NAMES = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
                   "m_total", "m_diff", "J_eff", "d"]
    M_TOTAL_IX = 6
    M_DIFF_IX = 7
    J_EFF_IX = 8
    M_DIFF_RHO = 0.99
    COORDINATE_VERSION = 2

    def __init__(self, combo_init=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        raw_init = self.params_init
        combo_init = to_tensor(combo_init).reshape(10).to(raw_init.dtype).clone()

        self.free_params = nn.Parameter(torch.zeros(10, dtype=raw_init.dtype))
        self.register_buffer("combo_init", combo_init)
        self.register_buffer(
            "Lambda_combo",
            torch.as_tensor(self._RMSE_baseline, dtype=raw_init.dtype) / combo_init.abs())

        r0 = combo_init[self.M_DIFF_IX] / self._m_diff_bound(combo_init)
        eta0 = torch.atanh(r0)
        scale = combo_init[self.M_DIFF_IX] / (self._m_diff_bound(combo_init) * (1 - r0 ** 2))
        self.register_buffer("coordinate_version", torch.as_tensor(
            self.COORDINATE_VERSION, dtype=torch.int64, device=combo_init.device))
        self.register_buffer("m_diff_eta0", eta0)
        self.register_buffer("m_diff_scale", scale)

    @staticmethod
    def combos_of(raw: Tensor, Lb) -> Tensor:
        """The ten combinations of a raw fourteen-vector, as a TENSOR (autograd safe)."""
        (kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d) = raw
        Lb = torch.as_tensor(Lb, dtype=raw.dtype, device=raw.device)
        return torch.stack([
            kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh,
            m1 + m2 + mb, m1 - m2,
            Jb + Jh + (m1 + m2) * Lb ** 2 / 4, d,
        ])

    @staticmethod
    def gauge_section(combo: Tensor, raw_init: Tensor, Lb) -> Tensor:
        """One raw fourteen-vector per combination ten-vector. A GAUGE FIXING, not an estimate."""
        kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d = combo
        Lb = torch.as_tensor(Lb, dtype=combo.dtype, device=combo.device)
        mb = raw_init[10]
        Jb0, Jh0 = raw_init[11], raw_init[12]
        m_sum = m_total - mb
        J_sum = J_eff - m_sum * Lb ** 2 / 4
        r = Jb0 / (Jb0 + Jh0)
        half = combo.new_tensor(0.5)
        return torch.stack([
            kb_sum * half, kb_sum * half, cg1, cg2, cy, cb_sum * half, cb_sum * half,
            mh, (m_sum + m_diff) / 2, (m_sum - m_diff) / 2, mb.expand_as(mh),
            J_sum * r, J_sum * (1 - r), d,
        ])

    def _m_diff_bound(self, combo: Tensor) -> Tensor:
        """`rho (2/Lb) sqrt(m_total J_eff)`: the contracted bound on `|m_diff|` (D-229)."""
        k = 2 * self.M_DIFF_RHO / self.Lb.to(combo.dtype)
        return k * torch.sqrt(combo[..., self.M_TOTAL_IX] * combo[..., self.J_EFF_IX])

    def combinations_from_free(self, free: Tensor) -> Tensor:
        """The ten combinations at free coordinate `free` (..., 10): nine log, `m_diff` bounded."""
        i = self.M_DIFF_IX
        c = self.combo_init * torch.exp(free)
        m_diff = self._m_diff_bound(c) * torch.tanh(
            self.m_diff_eta0 + self.m_diff_scale * free[..., i])
        return torch.cat((c[..., :i], m_diff.unsqueeze(-1), c[..., i + 1:]), dim=-1)

    def free_from_combinations(self, combo: Tensor) -> Tensor:
        """Inverse of `combinations_from_free` for admissible combinations (..., 10). Reporting only."""
        i = self.M_DIFF_IX
        c = torch.as_tensor(combo, dtype=self.combo_init.dtype, device=self.combo_init.device)
        positive = torch.cat((c[..., :i], c[..., i + 1:]), dim=-1)
        if not torch.isfinite(c).all() or (positive <= 0).any():
            raise ValueError("combinations must be finite and positive except for signed m_diff")
        r = c[..., i] / self._m_diff_bound(c)
        if (r.abs() >= 1).any():
            raise ValueError("m_diff lies outside the rho-contracted positive-definite domain")
        init = torch.cat((self.combo_init[:i], self.combo_init[i + 1:]))
        log = torch.log(positive / init)
        xi = (torch.atanh(r) - self.m_diff_eta0) / self.m_diff_scale
        return torch.cat((log[..., :i], xi.unsqueeze(-1), log[..., i:]), dim=-1)

    def recover_combinations(self) -> Tensor:
        """The ten combinations at the current free coordinate."""
        return self.combinations_from_free(self.free_params)

    # CHANGED (D-192): the transition as a PURE function of the free coordinate.
    def transition_from_free(self, free: Tensor, z: Tensor) -> Tensor:
        """`x_plus = f_base^d(v, x, u)` at the free coordinate `v = free`, without side effects."""
        assert z.size(1) == self.nx + self.nu
        p = self.gauge_section(self.combinations_from_free(free), self.params_init, self.Lb)
        mats = self._mats_from_raw(p)
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    # CHANGED (D-194): the ONCE-PER-OBJECTIVE half of the correction.
    def mats_and_tangent_from_free(self, free: Tensor, dfree: Tensor):
        """`(mats, dmats)` at the free coordinate `free`, differentiated in the direction `dfree`."""
        def _mats(v):
            return self._mats_from_raw(
                self.gauge_section(self.combinations_from_free(v), self.params_init, self.Lb))
        return torch.func.jvp(_mats, (free,), (dfree,))

    def _recover_params(self) -> Tensor:
        """Raw fourteen-vector in the declared gauge. Not clamped like the raw sibling."""
        return self.gauge_section(self.recover_combinations(), self.params_init, self.Lb)

    def estimation_parameters(self):
        """Names, initial values, and current values in identifiable coordinates."""
        return (tuple(self.COMBO_NAMES), self.combo_init.detach(),
                self.recover_combinations().detach(), "identifiable combinations")
