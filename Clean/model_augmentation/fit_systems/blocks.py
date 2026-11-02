"""The blocks of the augmented model: Jan's block types and the gantry physics blocks of the baseline.

Conventions: batch rows are samples, so A x is written x @ A.T; superscript n = normalised; polynomials in Y are
evaluated in Horner form. Code names: alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma_ = J_eff (the mass terms
in mats); a = qddot; fnet = u - C qdot - K q; free / free_params = beta, the training coordinates; M_DIFF_RHO = gamma,
the margin of the m_Delta bound.

Units: theta and theta_base are physical, beta is dimensionless. mats = (K, C, A_combined, m_h, alpha, beta, gamma_,
N0, N1, N2) with K, C, N0, N1, N2 (3, 3) and A_combined (6, 15); dmats has the same layout. One objective is one
evaluation of V or one validation; its matrices are built once (pass_mats).
"""
# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public,
# commit 8200923 (model_augmentation/__init__.py)
import torch
from torch import Tensor, nn

# CHANGED: the @added marker and gantry_ss for the added gantry blocks; unused imports removed
from model_augmentation.utils.utils import to_tensor, added
from model_augmentation.systems import gantry_ss


class Block(nn.Module):
    """Basic block implementation with variables that need to be defined for interconnect."""

    def __init__(self, nz, nw, **kwargs) -> None:
        """Store the block input size nz and output size nw."""
        super().__init__(**kwargs)

        self.nw = nw
        self.nz = nz


class Static_ANN_Block(Block):
    """Network block: (f_aug^n, g_aug^n) = phi_aug(x_hat^n, u^n), times a gate per output."""
    def __init__(self, nz, nw, net, n_nodes_per_layer, n_hidden_layers, activation, net_kwargs) -> None:
        """Network block: phi_aug with a gate per output.

        The interconnect feeds it z = (x_hat^n, u^n) and routes its outputs to the state rows.

        Args:
            nz: input size 6 + n_xbar + 3.
            nw: output size, one per routed state.
            net: network class (AugmentationNetwork).
            n_nodes_per_layer: width n_h of each hidden layer.
            n_hidden_layers: number of hidden layers.
            activation: tanh.
            net_kwargs: extra network arguments (nx_phys).
        """
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
        # CHANGED: a gate per output, 1 in training; the evaluation sets it to 0 to remove the network
        self.register_buffer('out_gate', torch.ones(self.nw), persistent=False)

    def forward(self, z: Tensor):
        """Network block of the augmented model.

        w = gate * phi_aug(z), z = (x_hat^n, u^n); the gate is 0 only when the evaluation removes the network.

        Args:
            z: (B, nz, 1) network input (x_tilde^n, xbar, u^n).
        Returns:
            (B, nw, 1) outputs (f_aug^n, g_aug^n), routed to the state rows by the interconnect.
        """
        w = self.net(z.view(-1, self.nz))
        w = w * self.out_gate
        return w.view(-1, self.nw, 1)


class Linear_Output_Block(Block):
    """Output map y_hat^n = C^n x_tilde^n + D u, fixed, in normalised coordinates; D must stay 0 for the closed loop."""
    # CHANGED: C and D required, as buffers; no empty-matrix cases
    def __init__(self, C, D) -> None:
        """C and D as fixed buffers; ny, nx, nu from their shapes."""
        _C = to_tensor(C)
        _D = to_tensor(D)

        self.ny = _C.size(0)
        self.nx = _C.size(1)
        self.nu = _D.size(1)

        super().__init__(nw=self.ny, nz=self.nx + self.nu)

        self.register_buffer("C", _C)
        self.register_buffer("D", _D)

    def forward(self, z: Tensor):
        """Output map of the augmented model.

        y_hat^n = C^n x_tilde^n + D u^n, D = 0, so the controller can use the output before the state update.

        Args:
            z: (B, nx + nu, 1) the selected physical state x_tilde^n and the input, normalised.
        Returns:
            (B, ny, 1) predicted output y_hat^n, normalised.
        """
        # CHANGED: no size assertion
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w


class Discrete_Nonlinear_Function_Block(Block):
    """w = nonlinear_function(z), defined by the subclass."""
    def forward(self, z: Tensor, mats):
        """Evaluate the subclass's map with the matrices of this objective."""
        # CHANGED: the per-objective matrices are passed on to nonlinear_function
        return self.nonlinear_function(z, mats)


@added
class GantryStateBlock(Discrete_Nonlinear_Function_Block):
    """Baseline transition f_base^n: one RK4 step over T_s of xdot = F(x, u), with M(Y)^-1 in rational form, in
    normalised coordinates."""

    def __init__(self, std_x, std_u, x_mean, u_mean, Ts, up_sample, *args, closed_form=False, **kwargs) -> None:
        """Baseline transition f_base^n.

        Stores the force transform P and the normalisation, so the block maps normalised states and inputs.

        Args:
            std_x, x_mean: (6, 1) sigma_x, mu_x of x_tilde = (q, qdot), physical.
            std_u, u_mean: (3, 1) sigma_u, mu_u of u_act = (F_X1, F_X2, F_Y) [N].
            Ts: sample time T_s [s].
            up_sample: RK4 steps per T_s.
            closed_form: xdot = (qdot, a) directly instead of through the LFR output equation.
        """
        super().__init__(nz=9, nw=6, *args, **kwargs)

        self.Ts   = Ts
        self.up_sample = up_sample
        self.nx   = 6
        self.closed_form = closed_form

        # 1. Force transform u = P u_act.
        # from (F_X1, F_X2, F_Y) to (F_X, torque on Theta, F_Y)
        self.register_buffer("P_mat", gantry_ss.P.clone())

        # 2. Normalisation buffers.
        self.register_buffer("std_x",    to_tensor(std_x).reshape(6, 1))
        self.register_buffer("std_x_1d", to_tensor(std_x).reshape(6))
        self.register_buffer("std_u",    to_tensor(std_u).reshape(3, 1))
        self.register_buffer("x_mean",   to_tensor(x_mean).reshape(6, 1))
        self.register_buffer("u_mean",   to_tensor(u_mean).reshape(3, 1))

    def _rk4(self, x: Tensor, u: Tensor, mats) -> Tensor:
        """Baseline transition f_base^n: one RK4 step of the vector field over T_s.

        x_{k+1} = x_k + (k1 + 2 k2 + 2 k3 + k4) / 6, h = T_s / up_sample, u held over the step.

        Args:
            x: (B, 6, 1) x_tilde^n_k.
            u: (B, 3, 1) u^n_k.
            mats: matrices of this objective.
        Returns:
            (B, 6, 1) x_tilde^n_{k+1} = f_base^n(theta_base, x_tilde^n_k, u^n_k).
        """
        h = self.Ts / self.up_sample
        for _ in range(self.up_sample):
            # stages: k1 = h F(x, u), k2 = h F(x + k1/2, u), k3 = h F(x + k2/2, u), k4 = h F(x + k3, u)
            k1 = h * self._deriv_with(x, u, mats)
            k2 = h * self._deriv_with(x + k1 / 2, u, mats)
            k3 = h * self._deriv_with(x + k2 / 2, u, mats)
            k4 = h * self._deriv_with(x + k3, u, mats)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def _deriv_with(self, x: Tensor, u: Tensor, mats) -> Tensor:
        """Baseline vector field F.

        xdot = (qdot, M(Y)^-1 (P u - K q - C qdot)), evaluated in physical units and returned normalised; the RK4 stages
        of f_base^n call it.

        Args:
            x: (B, 6, 1) x_tilde^n.
            u: (B, 3, 1) u^n.
            mats: matrices of this objective.
        Returns:
            (B, 6, 1) xdot^n = xdot / sigma_x [1/s].
        """
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats

        # 1. Denormalise: x = x^n sigma_x + mu_x, u = u^n sigma_u + mu_u.
        x_phys = x * self.std_x + self.x_mean
        u_phys = u * self.std_u + self.u_mean

        x2 = x_phys.squeeze(-1)
        u2 = u_phys.squeeze(-1)

        # 2. Net force.
        # generalised force u = P u_act
        u_log = u2 @ self.P_mat.T

        # fnet = u - K q - C qdot
        fnet = (-(x2[:, :3] @ K_mat.T)
                - (x2[:, 3:] @ C_mat.T)
                + u_log)

        # 3. Acceleration a = M(Y)^-1 fnet = (N0 + Y N1 + Y^2 N2) fnet / d(Y),
        # d(Y) = det M(Y) = m_h (alpha gamma - beta^2 + 2 beta m_h Y + m_h (alpha - m_h) Y^2); no linear solve per stage
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

        # 4. State derivative.
        if self.closed_form:
            # closed form: xdot = (qdot, a)
            xdot_phys = torch.cat([x2[:, 3:], a], dim=-1)
        else:
            # LFR output equation: xdot = A_combined (x, w, u), z = (a, Y a), w = Delta(Y) z = Y z
            z = torch.cat([a, Y_val * a], dim=-1)
            w = Y_val * z

            combined  = torch.cat([x2, w, u_log], dim=-1)
            xdot_phys = combined @ A_combined.T

        # 5. Normalise: xdot^n = xdot / sigma_x.
        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot

    def _deriv_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        """Baseline vector field F and its directional derivative.

        (F^n, DF^n[sx, dmats]): F^n of _deriv_with and its derivative along the state tangent sx and the matrix
        tangent dmats, written out in forward mode; the OBC correction uses it for varphi a.

        Args:
            x: (B, 6, 1) x_tilde^n.
            sx: (B, 6, 1) state tangent.
            u: (B, 3, 1) u^n.
            mats, dmats: matrices of this objective and their tangent.
        Returns:
            xdot^n and its tangent s_xdot^n, both (B, 6, 1) [1/s].
        """
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats
        (dK, dC, dA_combined,
         dmh, dalpha, dbeta, dgamma_, dN0, dN1, dN2) = dmats

        # every s_* below is the directional derivative of * along (sx, dmats)
        # 1. Denormalise: x = x^n sigma_x + mu_x, s_x = sx sigma_x; u = u^n sigma_u + mu_u, u = P u_act.
        s_x2 = (sx * self.std_x).squeeze(-1)
        x2 = (x * self.std_x + self.x_mean).squeeze(-1)
        u2 = (u * self.std_u + self.u_mean).squeeze(-1)
        u_log = u2 @ self.P_mat.T

        # 2. Net force and its tangent: fnet = u - K q - C qdot, s_fnet = -K s_q - dK q - C s_qdot - dC qdot.
        fnet = -(x2[:, :3] @ K_mat.T) - (x2[:, 3:] @ C_mat.T) + u_log
        s_fnet = (-(s_x2[:, :3] @ K_mat.T) - (x2[:, :3] @ dK.T)
                  - (s_x2[:, 3:] @ C_mat.T) - (x2[:, 3:] @ dC.T))

        # 3. Acceleration a = num / d.
        # determinant d(Y) = m_h poly(Y), poly = alpha gamma - beta^2 + 2 beta m_h Y + m_h (alpha - m_h) Y^2
        Y, sY = x2[:, 2], s_x2[:, 2]
        poly = (alpha * gamma_ - beta ** 2
                + 2 * beta * mh * Y
                + mh * (alpha - mh) * Y ** 2)
        s_poly = (dalpha * gamma_ + alpha * dgamma_ - 2 * beta * dbeta
                  + 2 * (dbeta * mh * Y + beta * dmh * Y + beta * mh * sY)
                  + (dmh * (alpha - mh) + mh * (dalpha - dmh)) * Y ** 2
                  + mh * (alpha - mh) * 2 * Y * sY)
        # s_d = dm_h poly + m_h s_poly
        dY_den = mh * poly
        s_dY_den = dmh * poly + mh * s_poly

        # numerator num = N0 fnet + Y (N1 fnet + Y N2 fnet) (Horner)
        Y_r, sY_r = Y.unsqueeze(0), sY.unsqueeze(0)
        n0f, n1f, n2f = N0 @ fnet.T, N1 @ fnet.T, N2 @ fnet.T
        # s_(N_j fnet) = dN_j fnet + N_j s_fnet; s_num by the product rule
        s_n0f = dN0 @ fnet.T + N0 @ s_fnet.T
        s_n1f = dN1 @ fnet.T + N1 @ s_fnet.T
        s_n2f = dN2 @ fnet.T + N2 @ s_fnet.T
        num = n0f + Y_r * (n1f + Y_r * n2f)
        s_num = (s_n0f + sY_r * (n1f + Y_r * n2f)
                 + Y_r * (s_n1f + sY_r * n2f + Y_r * s_n2f))

        # a = num / d, s_a = (s_num - a s_d) / d (quotient rule)
        a = num.T / dY_den[:, None]
        s_a = (s_num.T - a * s_dY_den[:, None]) / dY_den[:, None]

        # 4. State derivative.
        if self.closed_form:
            # closed form: (xdot, s_xdot) = ((qdot, a), (s_qdot, s_a))
            xdot_phys = torch.cat([x2[:, 3:], a], dim=-1)
            s_xdot_phys = torch.cat([s_x2[:, 3:], s_a], dim=-1)
        else:
            # LFR output equation, differentiated:
            # s_xdot = A_combined s_(x, w, u) + dA_combined (x, w, u), s_w = sY z + Y s_z, s_u = 0
            Y_val, sY_val = Y[:, None], sY[:, None]
            z = torch.cat([a, Y_val * a], dim=-1)
            s_z = torch.cat([s_a, sY_val * a + Y_val * s_a], dim=-1)
            w = Y_val * z
            s_w = sY_val * z + Y_val * s_z

            combined = torch.cat([x2, w, u_log], dim=-1)
            s_combined = torch.cat([s_x2, s_w, torch.zeros_like(u_log)], dim=-1)
            xdot_phys = combined @ A_combined.T
            s_xdot_phys = s_combined @ A_combined.T + combined @ dA_combined.T

        # 5. Normalise: xdot^n = xdot / sigma_x, likewise the tangent.
        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        s_xdot = (s_xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot, s_xdot

    def _rk4_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        """Baseline transition f_base^n and its directional derivative.

        The RK4 step of _rk4 and its tangent, stage by stage: k_i = h F(x_i), s_i = h DF(x_i)[sx_i, dmats]. With a zero
        state tangent and dmats along a, the tangent is varphi(beta_lin) a.

        Args:
            x, sx: (B, 6, 1) x_tilde^n and its tangent.
            u: (B, 3, 1) u^n.
            mats, dmats: matrices at beta_lin and their tangent.
        Returns:
            x_next and sx_next, both (B, 6, 1).
        """
        h = self.Ts / self.up_sample
        for _ in range(self.up_sample):
            # stages: k_i = h F(x_i), s_i = h DF(x_i)[sx_i, dmats], x_i and sx_i as in _rk4
            d1, e1 = self._deriv_with_tangent(x, sx, u, mats, dmats)
            k1, s1 = h * d1, h * e1
            d2, e2 = self._deriv_with_tangent(x + k1 / 2, sx + s1 / 2, u, mats, dmats)
            k2, s2 = h * d2, h * e2
            d3, e3 = self._deriv_with_tangent(x + k2 / 2, sx + s2 / 2, u, mats, dmats)
            k3, s3 = h * d3, h * e3
            d4, e4 = self._deriv_with_tangent(x + k3, sx + s3, u, mats, dmats)
            k4, s4 = h * d4, h * e4
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # tangent update: sx_next = sx + (s1 + 2 s2 + 2 s3 + s4) / 6
            sx = sx + (s1 + 2 * s2 + 2 * s3 + s4) / 6
        return x, sx


@added
class _TrainableGantryStateBlock(GantryStateBlock):
    """The baseline block whose matrices are rebuilt from the fourteen raw parameters theta."""

    # the order of theta; meanings and units at gantry_ss
    PARAM_NAMES = ("kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
                   "mh", "m1", "m2", "mb", "Jb", "Jh", "d")

    def __init__(self, *args, params_init, **kwargs) -> None:
        """theta^0 = params_init, and L_b fixed at its nominal value."""
        super().__init__(*args, **kwargs)

        params_init = to_tensor(params_init).reshape(14).clone()

        self.register_buffer("params_init", params_init)

        # L_b fixed at its nominal value, so q does not depend on the parameters
        self.register_buffer("Lb", gantry_ss.Lb.clone())

    def _build_KC(self, p: Tensor):
        """Damping and stiffness matrices.

        K = diag(0, k_b,Sigma, 0); C = [[c_g1 + c_g2, (L_b/2)(c_g1 - c_g2), 0], [(L_b/2)(c_g1 - c_g2),
        c_b,Sigma + (L_b^2/4)(c_g1 + c_g2), 0], [0, 0, c_y]]. The joint pairs enter only through their sums.

        Args:
            p: (14,) raw parameters theta.
        Returns:
            K and C, each (3, 3), physical.
        """
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         _, _, _, _, _, _, _) = p
        z = p.new_zeros(())
        Lb = self.Lb
        kb_sum, cb_sum = kb1 + kb2, cb1 + cb2
        # K = diag(0, k_b,Sigma, 0)
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
        """Y-dependent parts of the inertia matrix.

        M(Y) = M_0 + Y M_1 + Y^2 M_2: M_1 = -m_h (e1 e2^T + e2 e1^T), M_2 = m_h e2 e2^T.

        Args:
            mh: head mass m_h (scalar tensor).
        Returns:
            M_1 and M_2, each (3, 3).
        """
        z = mh.new_zeros(())
        # M_1 = -m_h (e1 e2^T + e2 e1^T)
        M1 = torch.stack([
            torch.stack([z,   -mh,  z]),
            torch.stack([-mh,  z,   z]),
            torch.stack([z,    z,   z]),
        ])
        # M_2 = m_h e2 e2^T
        M2 = torch.stack([
            torch.stack([z,   z,   z]),
            torch.stack([z,   mh,  z]),
            torch.stack([z,   z,   z]),
        ])
        return M1, M2

    def _mats_from_raw(self, p: Tensor):
        """Baseline matrices of one objective.

        theta -> (K, C, A_combined, m_h, alpha, beta, gamma_, N0, N1, N2), with M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / d(Y)
        and A_combined the state row of the LFR; built once per objective and reused by every RK4 stage.

        Args:
            p: (14,) raw parameters theta.
        Returns:
            the tuple above, physical.
        """
        (_, _, _, _, _, _, _,
         mh, m1, m2, mb, Jb, Jh, d) = p
        alpha, beta, gamma_, N0, N1, N2 = gantry_ss.build_poly_constants(
            m1, m2, mb, mh, Jb, Jh, self.Lb, d)
        # d0 = det M_0 = m_h (alpha gamma - beta^2), so M_0^-1 = N0 / d0
        d0 = mh * (alpha * gamma_ - beta ** 2)
        M1, M2 = self._build_M1M2(mh)
        K, C = self._build_KC(p)
        _, _, _, A_combined = gantry_ss.build_G_matrix_entries(N0, d0, M1, M2, K, C)
        return (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2)

    def nonlinear_function(self, z: Tensor, mats):
        """Baseline transition f_base^n as a block.

        x_tilde^n_{k+1} = f_base^n(theta_base, x_tilde^n_k, u^n_k); the interconnect adds f_aug^n to its output.

        Args:
            z: (B, 9, 1) (x_tilde^n_k, u^n_k).
            mats: matrices of this objective (pass_mats).
        Returns:
            (B, 6, 1) x_tilde^n_{k+1}.
        """
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    def pass_mats(self):
        """The matrices at the current beta, built once per objective."""
        return self._mats_at(self.free_params)


@added
class IdentifiableGantryStateBlock(_TrainableGantryStateBlock):
    """The baseline block trained in theta_base = (k_b,Sigma, c_g1, c_g2, c_y, c_b,Sigma, m_h, m_Sigma, m_Delta,
    J_eff, d) through training coordinates beta (free_params), beta = 0 at theta^0_base."""

    COMBO_NAMES = ("kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
                   "m_total", "m_diff", "J_eff", "d")
    M_TOTAL_IX = 6
    M_DIFF_IX = 7
    J_EFF_IX = 8
    # raw parameters the gauge section keeps from theta^0
    MB_IX = _TrainableGantryStateBlock.PARAM_NAMES.index('mb')
    JB_IX = _TrainableGantryStateBlock.PARAM_NAMES.index('Jb')
    JH_IX = _TrainableGantryStateBlock.PARAM_NAMES.index('Jh')
    # HEURISTIC: margin gamma = 0.99 of the |m_Delta| bound (_m_diff_bound)
    M_DIFF_RHO = 0.99
    # version of the map beta -> theta_base; raise it when the map changes, so older checkpoints are refused
    COORDINATE_VERSION = 2

    def __init__(self, *args, combo_init, **kwargs) -> None:
        """Baseline block in the training coordinates beta.

        beta = 0 at theta^0_base = combo_init; every beta gives an admissible baseline.

        Args:
            combo_init: (10,) theta^0_base, physical.
            *args, **kwargs: those of _TrainableGantryStateBlock.
        """
        super().__init__(*args, **kwargs)

        # 1. beta = 0 at theta^0_base.
        raw_init = self.params_init
        combo_init = to_tensor(combo_init).reshape(10).to(raw_init.dtype).clone()

        self.free_params = nn.Parameter(torch.zeros(10, dtype=raw_init.dtype))
        self.register_buffer("combo_init", combo_init)

        # 2. Calibrate the signed coordinate.
        # m_Delta map: r_0 = m_Delta,0 / b_0, eta_0 = atanh(r_0), s = m_Delta,0 / (b_0 (1 - r_0^2)),
        # so beta = 0 gives m_Delta,0 with slope m_Delta,0, like the nine log coordinates
        bound = self._m_diff_bound(combo_init)
        r0 = combo_init[self.M_DIFF_IX] / bound
        eta0 = torch.atanh(r0)
        scale = combo_init[self.M_DIFF_IX] / (bound * (1 - r0 ** 2))
        self.register_buffer("coordinate_version", torch.as_tensor(
            self.COORDINATE_VERSION, dtype=torch.int64, device=combo_init.device))
        self.register_buffer("m_diff_eta0", eta0)
        self.register_buffer("m_diff_scale", scale)

    @staticmethod
    def combos_of(raw: Tensor, Lb) -> Tensor:
        """Identifiable combinations theta_base(theta).

        k_b,Sigma = k_b1 + k_b2, c_b,Sigma = c_b1 + c_b2, m_Sigma = m_1 + m_2 + m_b, m_Delta = m_1 - m_2,
        J_eff = J_b + J_h + (L_b^2/4)(m_1 + m_2); the matrices depend on theta only through these ten.

        Args:
            raw: (14,) raw parameters theta.
            Lb: actuator distance L_b [m].
        Returns:
            (10,) theta_base in COMBO_NAMES order.
        """
        (kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d) = raw
        Lb = torch.as_tensor(Lb, dtype=raw.dtype, device=raw.device)
        return torch.stack([
            kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh,
            m1 + m2 + mb, m1 - m2,
            Jb + Jh + (m1 + m2) * Lb ** 2 / 4, d,
        ])

    @classmethod
    def gauge_section(cls, combo: Tensor, raw_init: Tensor, Lb) -> Tensor:
        """A raw parameter vector with given combinations.

        theta with theta_base(theta) = combo: equal splits of the sums, m_b and the ratio J_b : J_h from raw_init. A
        gauge choice, not an estimate: any such theta gives the same matrices.

        Args:
            combo: (10,) theta_base.
            raw_init: (14,) theta^0, the source of m_b and J_b : J_h.
            Lb: actuator distance L_b [m].
        Returns:
            (14,) theta.
        """
        kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d = combo
        Lb = torch.as_tensor(Lb, dtype=combo.dtype, device=combo.device)
        mb = raw_init[cls.MB_IX]
        Jb0, Jh0 = raw_init[cls.JB_IX], raw_init[cls.JH_IX]
        # m_1 + m_2 = m_Sigma - m_b, J_b + J_h = J_eff - (L_b^2/4)(m_1 + m_2), r = J_b0 / (J_b0 + J_h0)
        m_sum = m_total - mb
        J_sum = J_eff - m_sum * Lb ** 2 / 4
        r = Jb0 / (Jb0 + Jh0)
        return torch.stack([
            # equal splits of k_b,Sigma, c_b,Sigma; m_1,2 = (m_1 + m_2 +- m_Delta) / 2
            # J_b = r J_sum, J_h = (1 - r) J_sum
            kb_sum * 0.5, kb_sum * 0.5, cg1, cg2, cy, cb_sum * 0.5, cb_sum * 0.5,
            mh, (m_sum + m_diff) / 2, (m_sum - m_diff) / 2, mb.expand_as(mh),
            J_sum * r, J_sum * (1 - r), d,
        ])

    def _m_diff_bound(self, combo: Tensor) -> Tensor:
        """Bound on |m_Delta|.

        b = gamma (2 / L_b) sqrt(m_Sigma J_eff), gamma = M_DIFF_RHO = 0.99; |m_Delta| < b keeps
        m_Sigma J_eff > (L_b^2/4) m_Delta^2, so M(Y) > 0 for every Y.

        Args:
            combo: (..., 10) theta_base.
        Returns:
            (...,) bound b.
        """
        k = 2 * self.M_DIFF_RHO / self.Lb.to(combo.dtype)
        return k * torch.sqrt(combo[..., self.M_TOTAL_IX] * combo[..., self.J_EFF_IX])

    def combinations_from_free(self, free: Tensor) -> Tensor:
        """Parameter map theta_base(beta).

        theta_base,j = theta^0_base,j exp(beta_j) for the nine positive combinations, m_Delta = b tanh(eta_0 + s
        beta_mDelta); every beta gives an admissible baseline, so no bound or penalty is needed.

        Args:
            free: (..., 10) training coordinates beta.
        Returns:
            (..., 10) theta_base.
        """
        i = self.M_DIFF_IX
        # positive combinations: theta_base,j = theta^0_base,j exp(beta_j)
        c = self.combo_init * torch.exp(free)
        # signed mass difference: m_Delta = b tanh(eta_0 + s beta_mDelta)
        m_diff = self._m_diff_bound(c) * torch.tanh(
            self.m_diff_eta0 + self.m_diff_scale * free[..., i])
        return torch.cat((c[..., :i], m_diff.unsqueeze(-1), c[..., i + 1:]), dim=-1)

    def free_from_combinations(self, combo: Tensor) -> Tensor:
        """Inverse parameter map beta(theta_base).

        beta_j = log(theta_base,j / theta^0_base,j), beta_mDelta = (atanh(m_Delta / b) - eta_0) / s; the evaluation uses
        it for the coordinate of the true combinations.

        Args:
            combo: (..., 10) admissible theta_base.
        Returns:
            (..., 10) beta.
        """
        i = self.M_DIFF_IX
        c = torch.as_tensor(combo, dtype=self.combo_init.dtype, device=self.combo_init.device)
        positive = torch.cat((c[..., :i], c[..., i + 1:]), dim=-1)
        if not torch.isfinite(c).all() or (positive <= 0).any():
            raise ValueError("combinations must be finite and positive except for signed m_diff; pass "
                             "physical combinations")
        # r = m_Delta / b, |r| < 1 required
        r = c[..., i] / self._m_diff_bound(c)
        if (r.abs() >= 1).any():
            raise ValueError("m_diff lies outside the rho-contracted positive-definite domain, so no training "
                             "coordinate represents it; pass admissible combinations")
        init = torch.cat((self.combo_init[:i], self.combo_init[i + 1:]))
        # beta_j = log(theta_base,j / theta^0_base,j), beta_mDelta = (atanh(r) - eta_0) / s
        log = torch.log(positive / init)
        xi = (torch.atanh(r) - self.m_diff_eta0) / self.m_diff_scale
        return torch.cat((log[..., :i], xi.unsqueeze(-1), log[..., i:]), dim=-1)

    def recover_combinations(self) -> Tensor:
        """Current parameter estimate theta_base(beta), for reporting.

        Returns:
            (10,) theta_base, physical.
        """
        return self.combinations_from_free(self.free_params)

    def transition_from_free(self, free: Tensor, z: Tensor) -> Tensor:
        """Baseline transition as a function of beta.

        f_base^n(theta_base(beta), x_tilde^n, u^n), without changing the block; the OBC basis differentiates it.

        Args:
            free: (10,) beta.
            z: (B, 9, 1) (x_tilde^n, u^n).
        Returns:
            (B, 6, 1) next physical state.
        """
        mats = self._mats_at(free)
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    def mats_and_tangent_from_free(self, free: Tensor, dfree: Tensor):
        """Baseline matrices and their directional derivative in beta.

        (m(beta), Dm(beta)[dfree]) by forward-mode jvp; with dfree = a this gives the matrix tangent of varphi a.

        Args:
            free: (10,) beta (beta_lin for the OBC correction).
            dfree: (10,) direction a.
        Returns:
            The matrix tuple and its tangent tuple.
        """
        return torch.func.jvp(self._mats_at, (free,), (dfree,))

    def _mats_at(self, free: Tensor):
        """Matrices at beta = free: beta -> theta_base -> theta (gauge_section) -> matrices."""
        return self._mats_from_raw(
            self.gauge_section(self.combinations_from_free(free), self.params_init, self.Lb))

    def estimation_parameters(self):
        """Names, start values and current values of the identifiable combinations."""
        return self.COMBO_NAMES, self.combo_init.detach(), self.recover_combinations().detach()
