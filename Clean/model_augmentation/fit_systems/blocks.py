"""The blocks of the augmented model: Jan's block types and the gantry physics blocks of the baseline."""
# CHANGED: restricted to the thesis training path; full implementation in
# model_augmentation/fit_systems/blocks.py at the repository root
import torch
from torch import Tensor, nn

# CHANGED: explicit names instead of the star import, and gantry_ss for the added gantry blocks
from model_augmentation.utils.utils import to_tensor, added
from model_augmentation.systems import gantry_ss


class Block(nn.Module):
    """Basic block implementation with variables that need to be defined for interconnect."""

    def __init__(self, nz, nw, **kwargs) -> None:
        super().__init__(**kwargs)

        self.nw = nw
        self.nz = nz


class Static_ANN_Block(Block):
    """w = net(z), times a gate per output: the network block of the augmented model."""
    def __init__(self, nz, nw, net, n_nodes_per_layer, n_hidden_layers, activation, net_kwargs) -> None:
        super(Static_ANN_Block, self).__init__(nz, nw)

        # CHANGED: net_kwargs reach the network's constructor (the number of physical rows)
        self.net = net(
            n_in=self.nz,
            n_out=self.nw,
            n_nodes_per_layer=n_nodes_per_layer,
            n_hidden_layers=n_hidden_layers,
            activation=activation,
            **net_kwargs,
        )
        # CHANGED: a gate per output, 1 in training, applied in forward; _without_learned zeroes it to remove the
        # network (persistent=False: not in the state_dict)
        self.register_buffer('out_gate', torch.ones(self.nw), persistent=False)

    def forward(self, z: Tensor):
        """Apply the network and its per-output gate, returning (batch, nw, 1)."""
        w = self.net(z.view(-1, self.nz))
        w = w * self.out_gate
        return w.view(-1, self.nw, 1)


@added
class Linear_Output_Block(Block):
    """w = C x + D u, with C and D as buffers: they follow .to() and are not trained."""
    def __init__(self, C, D) -> None:
        _C = to_tensor(C)
        _D = to_tensor(D)

        self.ny = _C.size(0)
        self.nx = _C.size(1)
        self.nu = _D.size(1)

        super().__init__(nw=self.ny, nz=self.nx + self.nu)

        self.register_buffer("C", _C)
        self.register_buffer("D", _D)

    def forward(self, z: Tensor):
        """Return C x + D u from the concatenated state and input."""
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w


class Discrete_Nonlinear_Function_Block(Block):
    """w = nonlinear_function(z), defined by the subclass."""
    def forward(self, z: Tensor, mats=None):
        """Evaluate the subclass's nonlinear map with optional per-objective matrices."""
        # CHANGED: mats, the matrices built once per objective, are passed on to nonlinear_function
        return self.nonlinear_function(z, mats)


@added
class GantryStateBlock(Discrete_Nonlinear_Function_Block):
    """The gantry LPV-LFR baseline (Drenth et al.): the continuous-time model with M(Y)^-1 in rational form,
    integrated by RK4 over one sample in normalised coordinates."""

    def __init__(self, Y_op, std_x, std_u, x_mean, u_mean, Ts, up_sample, *args, **kwargs) -> None:
        super().__init__(nz=9, nw=6, *args, **kwargs)

        self.Ts   = Ts
        self.up_sample = up_sample
        self.nx   = 6
        self.Y_op = Y_op

        # the nominal LFR matrices of gantry_ss; the trainable subclasses rebuild them at every objective
        _alpha, _beta, _gamma, _N0, _N1, _N2 = gantry_ss.build_poly_constants(
            gantry_ss.m1, gantry_ss.m2, gantry_ss.mb, gantry_ss.mh,
            gantry_ss.Jb, gantry_ss.Jh, gantry_ss.Lb, gantry_ss.d
        )
        _d0 = gantry_ss.mh * (_alpha * _gamma - _beta ** 2)

        _Ax, _Bw, _Bu, _A_combined = gantry_ss.build_G_matrix_entries(
            _N0, _d0, gantry_ss.M1, gantry_ss.M2, gantry_ss.K, gantry_ss.C)

        self.register_buffer("mh",     gantry_ss.mh.clone())
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

        self.register_buffer("K_mat", gantry_ss.K.clone())
        self.register_buffer("C_mat", gantry_ss.C.clone())
        self.register_buffer("P_mat", gantry_ss.P.clone())

        self.register_buffer("std_x",    to_tensor(std_x).reshape(6, 1))
        self.register_buffer("std_x_1d", to_tensor(std_x).reshape(6))
        self.register_buffer("std_u",    to_tensor(std_u).reshape(3, 1))
        self.register_buffer("x_mean",   to_tensor(x_mean).reshape(6, 1))
        self.register_buffer("u_mean",   to_tensor(u_mean).reshape(3, 1))

    def _rk4(self, x: Tensor, u: Tensor, mats) -> Tensor:
        """up_sample classical RK4 steps of h = Ts / up_sample, with u held."""
        h = self.Ts / self.up_sample
        for i in range(self.up_sample):
            k1 = h * self._deriv_with(x, u, mats)
            k2 = h * self._deriv_with(x + k1 / 2, u, mats)
            k3 = h * self._deriv_with(x + k2 / 2, u, mats)
            k4 = h * self._deriv_with(x + k3, u, mats)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def _deriv_with(self, x: Tensor, u: Tensor, mats) -> Tensor:
        """xdot^n at (x^n, u^n) with the matrices mats."""
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats

        # physical units inside: x = x^n sigma_x + x_mean, u = u^n sigma_u + u_mean
        x_phys = x * self.std_x + self.x_mean
        u_phys = u * self.std_u + self.u_mean

        x2 = x_phys.squeeze(-1)
        u2 = u_phys.squeeze(-1)

        u_log = u2 @ self.P_mat.T

        # f = P u - K q - C qdot
        fnet = (-(x2[:, :3] @ K_mat.T)
                - (x2[:, 3:] @ C_mat.T)
                + u_log)

        if self.Y_op is not None:
            # frozen scheduling, set by the evaluation: a = N_op f / d_op at Y = Y_op
            Y_val = self.Y_op
            a = (self.N_op @ fnet.T).T / self.d_op
        else:
            # a = qddot = M(Y)^-1 f = (N0 + Y N1 + Y^2 N2) f / d(Y)
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

        # LFR: w = Y z, z = (a, Y a), xdot = Ax x + Bw w + Bu P u = A_combined (x, w, P u)
        z = torch.cat([a, Y_val * a], dim=-1)
        w = Y_val * z

        combined  = torch.cat([x2, w, u_log], dim=-1)
        xdot_phys = combined @ A_combined.T

        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot

    def _deriv_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        """Return xdot^n and its directional derivative along the state tangent sx and matrix tangent dmats."""
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats
        (dK, dC, dA_combined,
         dmh, dalpha, dbeta, dgamma_, dN0, dN1, dN2) = dmats

        if self.Y_op is not None:
            raise NotImplementedError(
                'the tangent covers the scheduled branch (Y_op=None) only; clear Y_op before using the OBC '
                'correction')

        # every s_* below is the directional derivative of * along (sx, dmats)
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

    def _rk4_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        """_rk4 for the state and its tangent together; returns (x_next, sx_next)."""
        h = self.Ts / self.up_sample
        for _ in range(self.up_sample):
            d1, e1 = self._deriv_with_tangent(x, sx, u, mats, dmats)
            k1, s1 = h * d1, h * e1
            d2, e2 = self._deriv_with_tangent(x + k1 / 2, sx + s1 / 2, u, mats, dmats)
            k2, s2 = h * d2, h * e2
            d3, e3 = self._deriv_with_tangent(x + k2 / 2, sx + s2 / 2, u, mats, dmats)
            k3, s3 = h * d3, h * e3
            d4, e4 = self._deriv_with_tangent(x + k3, sx + s3, u, mats, dmats)
            k4, s4 = h * d4, h * e4
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            sx = sx + (s1 + 2 * s2 + 2 * s3 + s4) / 6
        return x, sx


@added
class _TrainableGantryStateBlock(GantryStateBlock):
    """The gantry block with trainable parameters: its matrices are rebuilt from a raw fourteen-vector."""

    PARAM_NAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
                   "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]

    def __init__(self, RMSE_baseline: float = 1.0, params_init=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        params_init = to_tensor(params_init).reshape(14).clone()

        self.register_buffer("params_init", params_init)
        # the scale of the subclass's Lambda_combo buffer
        self._RMSE_baseline = float(RMSE_baseline)

        self.register_buffer("Lb", gantry_ss.Lb.clone())

    def _build_KC(self, p: Tensor):
        """Build K and C; the joint stiffness and damping pairs enter only through kb1 + kb2 and cb1 + cb2."""
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
        """M1, M2 of M(Y) = M0 + Y M1 + Y^2 M2; they depend on mh only."""
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
        """The matrices of one objective at a raw vector p: (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2)."""
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         mh, m1, m2, mb, Jb, Jh, d) = p
        alpha, beta, gamma_, N0, N1, N2 = gantry_ss.build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, self.Lb, d)
        d0 = mh * (alpha * gamma_ - beta ** 2)
        M1, M2 = self._build_M1M2(mh)
        K, C = self._build_KC(p)
        _, _, _, A_combined = gantry_ss.build_G_matrix_entries(N0, d0, M1, M2, K, C)
        return (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2)

    def nonlinear_function(self, z: Tensor, mats=None):
        """One sample of the transition; mats are the matrices of this objective, else rebuilt here."""
        if mats is None:
            mats = self._mats_at(self.free_params)
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    def pass_mats(self):
        """The matrices at the current parameters, built once per objective."""
        return self._mats_at(self.free_params)


@added
class IdentifiableGantryStateBlock(_TrainableGantryStateBlock):
    """The gantry block trained in the ten identifiable combinations, through training coordinates beta
    (free_params) that are zero at the start."""

    COMBO_NAMES = ["kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
                   "m_total", "m_diff", "J_eff", "d"]
    M_TOTAL_IX = 6
    M_DIFF_IX = 7
    J_EFF_IX = 8
    # HEURISTIC: m_diff stays within 0.99 of its admissible bound, see _m_diff_bound
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

        # m_diff = bound tanh(eta0 + scale beta_m): beta_m = 0 gives combo_init with slope m_diff,0, as the log
        # coordinates of the other nine have
        r0 = combo_init[self.M_DIFF_IX] / self._m_diff_bound(combo_init)
        eta0 = torch.atanh(r0)
        scale = combo_init[self.M_DIFF_IX] / (self._m_diff_bound(combo_init) * (1 - r0 ** 2))
        self.register_buffer("coordinate_version", torch.as_tensor(
            self.COORDINATE_VERSION, dtype=torch.int64, device=combo_init.device))
        self.register_buffer("m_diff_eta0", eta0)
        self.register_buffer("m_diff_scale", scale)

    @staticmethod
    def combos_of(raw: Tensor, Lb) -> Tensor:
        """The ten combinations of a raw fourteen-vector, as a tensor (autograd safe)."""
        (kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d) = raw
        Lb = torch.as_tensor(Lb, dtype=raw.dtype, device=raw.device)
        return torch.stack([
            kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh,
            m1 + m2 + mb, m1 - m2,
            Jb + Jh + (m1 + m2) * Lb ** 2 / 4, d,
        ])

    @staticmethod
    def gauge_section(combo: Tensor, raw_init: Tensor, Lb) -> Tensor:
        """One raw fourteen-vector with the given combinations, a gauge fixing and not an estimate: equal splits
        of the sums, mb and the ratio Jb : Jh from raw_init."""
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
        """rho (2 / Lb) sqrt(m_total J_eff), the contracted bound on |m_diff|."""
        k = 2 * self.M_DIFF_RHO / self.Lb.to(combo.dtype)
        return k * torch.sqrt(combo[..., self.M_TOTAL_IX] * combo[..., self.J_EFF_IX])

    def combinations_from_free(self, free: Tensor) -> Tensor:
        """The ten combinations at training coordinate free (..., 10): c0 exp(beta) for nine, m_diff by tanh."""
        i = self.M_DIFF_IX
        c = self.combo_init * torch.exp(free)
        m_diff = self._m_diff_bound(c) * torch.tanh(
            self.m_diff_eta0 + self.m_diff_scale * free[..., i])
        return torch.cat((c[..., :i], m_diff.unsqueeze(-1), c[..., i + 1:]), dim=-1)

    def free_from_combinations(self, combo: Tensor) -> Tensor:
        """The training coordinate of admissible combinations (..., 10), the inverse of combinations_from_free."""
        i = self.M_DIFF_IX
        c = torch.as_tensor(combo, dtype=self.combo_init.dtype, device=self.combo_init.device)
        positive = torch.cat((c[..., :i], c[..., i + 1:]), dim=-1)
        if not torch.isfinite(c).all() or (positive <= 0).any():
            raise ValueError("combinations must be finite and positive except for signed m_diff; pass "
                             "physical combinations")
        r = c[..., i] / self._m_diff_bound(c)
        if (r.abs() >= 1).any():
            raise ValueError("m_diff lies outside the rho-contracted positive-definite domain, so no training "
                             "coordinate represents it; pass admissible combinations")
        init = torch.cat((self.combo_init[:i], self.combo_init[i + 1:]))
        log = torch.log(positive / init)
        xi = (torch.atanh(r) - self.m_diff_eta0) / self.m_diff_scale
        return torch.cat((log[..., :i], xi.unsqueeze(-1), log[..., i:]), dim=-1)

    def recover_combinations(self) -> Tensor:
        """The ten combinations at the current training coordinate."""
        return self.combinations_from_free(self.free_params)

    def transition_from_free(self, free: Tensor, z: Tensor) -> Tensor:
        """Return the normalised baseline transition at beta = free without changing the block."""
        mats = self._mats_at(free)
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    def mats_and_tangent_from_free(self, free: Tensor, dfree: Tensor):
        """(matrices, their directional derivative along dfree) at training coordinate free."""
        return torch.func.jvp(self._mats_at, (free,), (dfree,))

    def _mats_at(self, free: Tensor):
        """The matrices at training coordinate free, from the raw vector of gauge_section."""
        return self._mats_from_raw(
            self.gauge_section(self.combinations_from_free(free), self.params_init, self.Lb))

    def estimation_parameters(self):
        """Names, start values and current values of the identifiable combinations."""
        return (tuple(self.COMBO_NAMES), self.combo_init.detach(),
                self.recover_combinations().detach(), "identifiable combinations")
