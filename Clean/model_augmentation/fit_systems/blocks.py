"""Blocks of the augmented model: the baseline predicts the next physical state, the network adds its correction
and predicts the next added states, and the output map gives the measured positions.

Jan Hoekstra's block types are adapted for this model; the gantry blocks are added. Superscript n = normalised;
theta holds fourteen physical parameters, theta_base their ten identifiable combinations, and beta the
dimensionless training coordinates. In shapes, ... denotes any leading batch dimensions.
"""
# CHANGED: restricted to the thesis training path; Jan's full code is in Model-Augmentation-Public-main
import torch
from torch import Tensor, nn

# CHANGED: the @added marker and gantry_ss for the added gantry blocks; unused imports removed
from model_augmentation.utils.utils import to_tensor, added
from model_augmentation.systems import gantry_ss


class Block(nn.Module):
    """Block input and output sizes nz and nw, used by the interconnect to connect its signals."""

    def __init__(self, nz, nw, **kwargs) -> None:
        """Store the block input size nz and output size nw."""
        # CHANGED: no name, no block_ix default, no init_block (a no-op) or abstract forward; add_block sets block_ix
        super().__init__(**kwargs)

        self.nw = nw
        self.nz = nz


class Static_ANN_Block(Block):
    """Network block phi_aug: physical-state corrections f_aug^n and next added states g_aug^n, one gate per output."""
    # CHANGED: nz and nw first, every argument required, no defaults, so every setting comes from build_model
    def __init__(self, nz, nw, net, n_nodes_per_layer, n_hidden_layers, activation, net_kwargs) -> None:
        """Network phi_aug: corrections f_aug^n to the physical states and next added states g_aug^n.

        The interconnect feeds it (x_hat^n, u^n), adds f_aug^n to the baseline transition and sets the next added
        states to g_aug^n.

        Args:
            nz: input size 6 + n_xbar + 3.
            nw: output size 6 + n_xbar, physical rows first.
            net: network class (AugmentationNetwork).
            n_nodes_per_layer: width n_h of each hidden layer.
            n_hidden_layers: number of hidden layers.
            activation: hidden-layer activation (nn.Tanh).
            net_kwargs: extra network arguments; nx_phys sets the number of physical output rows.
        """
        super(Static_ANN_Block, self).__init__(nz, nw)

        # CHANGED: net_kwargs reach the network's constructor: nx_phys, so it can zero the f_aug^n rows at the start
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
        """Network output w = out_gate * phi_aug(z); evaluation sets out_gate to zero to simulate the baseline alone.

        Args:
            z: (B, nz, 1) network input (x_tilde^n, xbar, u^n).
        Returns:
            (B, nw, 1) physical-state corrections followed by next added states.
        """
        # CHANGED: no size assertion; init_connection_matrices checks the sizes once
        w = self.net(z.view(-1, self.nz))
        w = w * self.out_gate
        return w.view(-1, self.nw, 1)


class Linear_Output_Block(Block):
    """Fixed output map y_hat^n = C^n x_tilde^n + D u^n; D = 0, so the output depends only on the physical state."""
    # CHANGED: C and D required, as buffers so they move to the GPU with the model; no empty-matrix cases
    def __init__(self, C, D) -> None:
        """Store C (ny, nx) and D (ny, nu) as buffers: they move with the model and are not trained."""
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
        # CHANGED: no size assertion; init_connection_matrices checks the sizes once
        x = z[:, : self.nx, :]
        u = z[:, self.nx :, :]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w


class Discrete_Nonlinear_Function_Block(Block):
    """Block whose subclass defines w = nonlinear_function(z, mats)."""
    def forward(self, z: Tensor, mats):
        """Evaluate the subclass's map with the matrices of this objective."""
        # CHANGED: the matrices, built once per objective, are passed on to nonlinear_function; no size assertion
        return self.nonlinear_function(z, mats)


@added
class GantryStateBlock(Discrete_Nonlinear_Function_Block):
    """Baseline transition f_base^n: RK4 over one sampling interval T_s, with up_sample substeps.

    The dynamics are computed in physical units; the block receives and returns normalised states.
    """

    def __init__(self, std_x, std_u, x_mean, u_mean, Ts, up_sample, closed_form=False) -> None:
        """Store the sampling interval, force transform P and normalisation of the baseline.

        Args:
            std_x, x_mean: (6, 1) sigma_x, mu_x of x_tilde = (q, qdot), physical.
            std_u, u_mean: (3, 1) sigma_u, mu_u of u_act = (F_X1, F_X2, F_Y) [N].
            Ts: sample time T_s [s].
            up_sample: RK4 steps per T_s.
            closed_form: compute xdot = (qdot, qddot) directly; otherwise use the equivalent LFR state equation.
        """
        super().__init__(nz=9, nw=6)

        self.Ts   = Ts
        self.up_sample = up_sample
        self.nx   = 6
        self.closed_form = closed_form

        # 1. Force transform u = P u_act.
        # from (F_X1, F_X2, F_Y) to (F_X, torque on Theta, F_Y)
        self.register_buffer("P_mat", gantry_ss.P.clone())

        # 2. Means and standard deviations; std_x_1d is for arrays with one state per row.
        self.register_buffer("std_x",    to_tensor(std_x).reshape(6, 1))
        self.register_buffer("std_x_1d", to_tensor(std_x).reshape(6))
        self.register_buffer("std_u",    to_tensor(std_u).reshape(3, 1))
        self.register_buffer("x_mean",   to_tensor(x_mean).reshape(6, 1))
        self.register_buffer("u_mean",   to_tensor(u_mean).reshape(3, 1))

    def _rk4(self, x: Tensor, u: Tensor, mats) -> Tensor:
        """Baseline transition f_base^n over T_s: up_sample RK4 substeps of length h = T_s / up_sample.

        The input is held constant over T_s; each substep adds (k1 + 2 k2 + 2 k3 + k4) / 6 to the state.

        Args:
            x: (B, 6, 1) x_tilde^n_k.
            u: (B, 3, 1) u^n_k.
            mats: matrices of this objective.
        Returns:
            (B, 6, 1) next normalised physical state.
        """
        h = self.Ts / self.up_sample
        for _ in range(self.up_sample):
            # stages, F = _deriv_with: k1 = h F(x, u), k2 = h F(x + k1/2, u), k3 = h F(x + k2/2, u), k4 = h F(x + k3, u)
            k1 = h * self._deriv_with(x, u, mats)
            k2 = h * self._deriv_with(x + k1 / 2, u, mats)
            k3 = h * self._deriv_with(x + k2 / 2, u, mats)
            k4 = h * self._deriv_with(x + k3, u, mats)
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def _deriv_with(self, x: Tensor, u: Tensor, mats) -> Tensor:
        """State derivative used by each RK4 stage: xdot = (qdot, M(Y)^-1 (P u_act - K q - C qdot)).

        State and actuator force are first denormalised; the derivative is divided by sigma_x on return.

        Args:
            x: (B, 6, 1) x_tilde^n.
            u: (B, 3, 1) u^n.
            mats: matrices of this objective.
        Returns:
            (B, 6, 1) normalised state derivative xdot^n [1/s].
        """
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats

        # 1. Denormalise: x = x^n sigma_x + mu_x, u = u^n sigma_u + mu_u.
        x_phys = x * self.std_x + self.x_mean
        u_phys = u * self.std_u + self.u_mean

        x2 = x_phys.squeeze(-1)
        u2 = u_phys.squeeze(-1)

        # 2. Actuator forces mapped to generalised forces (F_X, torque on Theta, F_Y); samples are rows.
        u_log = u2 @ self.P_mat.T

        # net generalised force: fnet = P u_act - K q - C qdot
        fnet = (-(x2[:, :3] @ K_mat.T)
                - (x2[:, 3:] @ C_mat.T)
                + u_log)

        # 3. Acceleration a = M(Y)^-1 fnet; the adjugate N0 + Y N1 + Y^2 N2 avoids a linear solve at each stage.
        # In these inertia terms beta = (L_b/2) m_Delta and gamma_ = J_eff; beta here is not the training coordinate.
        Y   = x2[:, 2]
        dY  = gantry_ss.inertia_determinant(Y, alpha, beta, gamma_, mh)
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
            # LFR state equation: xdot = A_combined (x, w, u), with w = (Y a, Y^2 a).
            z = torch.cat([a, Y_val * a], dim=-1)
            w = Y_val * z

            combined  = torch.cat([x2, w, u_log], dim=-1)
            xdot_phys = combined @ A_combined.T

        # 5. Normalise: xdot^n = xdot / sigma_x.
        xdot = (xdot_phys / self.std_x_1d).unsqueeze(-1)
        return xdot

    def _deriv_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        """State derivative and its directional derivative along (sx, dmats), with u fixed.

        Args:
            x: (B, 6, 1) x_tilde^n.
            sx: (B, 6, 1) derivative of the normalised state in the chosen direction.
            u: (B, 3, 1) u^n.
            mats, dmats: baseline matrices and their derivatives in the same direction.
        Returns:
            xdot^n and its directional derivative s_xdot^n, both (B, 6, 1) [1/s].
        """
        (K_mat, C_mat, A_combined,
         mh, alpha, beta, gamma_, N0, N1, N2) = mats
        (dK, dC, dA_combined,
         dmh, dalpha, dbeta, dgamma_, dN0, dN1, dN2) = dmats

        # s_* denotes a directional derivative; dK, dC, etc. are the matrix derivatives supplied in dmats.
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
            # LFR state equation, differentiated:
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

    def rk4_with_tangent(self, x: Tensor, sx: Tensor, u: Tensor, mats, dmats):
        """Baseline transition and its derivative in a chosen parameter direction, through every RK4 substep.

        OBC uses sx = 0 because its reference states are fixed; with dmats in direction a at beta_lin, the
        returned derivative is varphi(beta_lin) a.

        Args:
            x, sx: (B, 6, 1) normalised physical state and its derivative in the chosen direction.
            u: (B, 3, 1) u^n.
            mats, dmats: baseline matrices and their derivatives in the same direction.
        Returns:
            next normalised physical state and its directional derivative, both (B, 6, 1).
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
    """Build the baseline matrices from fourteen physical parameters theta and use them in the state transition."""

    # the order of theta; meanings and units at gantry_ss
    PARAM_NAMES = ("kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
                   "mh", "m1", "m2", "mb", "Jb", "Jh", "d")

    def __init__(self, *args, params_init, **kwargs) -> None:
        """Store the nominal physical parameters theta^0 = params_init; L_b stays fixed during training."""
        super().__init__(*args, **kwargs)

        params_init = to_tensor(params_init).reshape(14).clone()

        self.register_buffer("params_init", params_init)

        # L_b fixed at its nominal value, so q does not depend on the parameters
        self.register_buffer("Lb", gantry_ss.Lb.clone())

    def _build_KC(self, p: Tensor):
        """Stiffness K and damping C of the parameters p, as built in gantry_ss.stiffness_and_damping.

        Args:
            p: (14,) parameters theta.
        Returns:
            Stiffness K and damping C, each (3, 3).
        """
        (kb1, kb2, cg1, cg2, cy, cb1, cb2,
         _, _, _, _, _, _, _) = p
        return gantry_ss.stiffness_and_damping(kb1, kb2, cg1, cg2, cy, cb1, cb2, self.Lb)

    def _mats_from_parameters(self, p: Tensor):
        """Build stiffness, damping and the inertia terms used by the baseline dynamics.

        M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / det M(Y); A_combined maps (x, w, u) to xdot in the LFR.
        The matrices are reused by every RK4 stage of the rollout.

        Args:
            p: (14,) parameters theta.
        Returns:
            (K, C, A_combined, m_h, alpha, beta, gamma_, N0, N1, N2), in physical units:
            K, C, N0, N1, N2 each (3, 3), A_combined (6, 15), and scalar inertia terms
            alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma_ = J_eff;
            this beta is an inertia term, not the trained beta.
        """
        (_, _, _, _, _, _, _,
         mh, m1, m2, mb, Jb, Jh, d) = p
        alpha, beta, gamma_ = gantry_ss.inertia_constants(m1, m2, mb, mh, Jb, Jh, self.Lb)
        N0, N1, N2 = gantry_ss.inertia_adjugate(alpha, beta, gamma_, mh, d)
        d0 = gantry_ss.inertia_determinant(0.0, alpha, beta, gamma_, mh)
        M1, M2 = gantry_ss.inertia_parts(mh)
        K, C = self._build_KC(p)
        A_combined = gantry_ss.lfr_state_matrix(N0 / d0, M1, M2, K, C)
        return (K, C, A_combined, mh, alpha, beta, gamma_, N0, N1, N2)

    def nonlinear_function(self, z: Tensor, mats):
        """Baseline transition f_base^n as a block.

        x_tilde^n_{k+1} = f_base^n(theta_base, x_tilde^n_k, u^n_k); the interconnect adds f_aug^n to its output.

        Args:
            z: (B, 9, 1) (x_tilde^n_k, u^n_k).
            mats: matrices of this objective (pass_mats).
        Returns:
            (B, 6, 1) next normalised physical state.
        """
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    def pass_mats(self):
        """The matrices at the current beta, built once per objective."""
        return self._mats_at(self.beta)


@added
class IdentifiableGantryStateBlock(_TrainableGantryStateBlock):
    """Train the ten identifiable combinations theta_base using dimensionless coordinates beta."""

    COMBO_NAMES = ("kb_sum", "cg1", "cg2", "cy", "cb_sum", "mh",
                   "m_total", "m_diff", "J_eff", "d")
    M_TOTAL_IX = 6
    M_DIFF_IX = 7
    J_EFF_IX = 8
    # nominal cross-arm mass and inertias used to reconstruct fourteen parameters from the ten combinations
    MB_IX = _TrainableGantryStateBlock.PARAM_NAMES.index('mb')
    JB_IX = _TrainableGantryStateBlock.PARAM_NAMES.index('Jb')
    JH_IX = _TrainableGantryStateBlock.PARAM_NAMES.index('Jh')
    # HEURISTIC: margin rho = 0.99 of the |m_Delta| bound (_m_diff_bound)
    M_DIFF_RHO = 0.99
    # version of the map beta -> theta_base; raise it when the map changes, so older checkpoints are refused
    COORDINATE_VERSION = 2

    def __init__(self, *args, combo_init, **kwargs) -> None:
        """Set beta = 0 at combo_init and calibrate the signed mass-difference coordinate.

        Args:
            combo_init: (10,) starting combinations theta^0_base, nominal or detuned, in physical units.
            *args, **kwargs: those of _TrainableGantryStateBlock.
        """
        super().__init__(*args, **kwargs)

        # 1. beta = 0 at theta^0_base.
        theta_init = self.params_init
        combo_init = to_tensor(combo_init).reshape(10).to(theta_init.dtype).clone()

        # keep the saved parameter name, so existing checkpoints and Adam moments still load
        self.register_parameter('free_params', nn.Parameter(torch.zeros(10, dtype=theta_init.dtype)))
        self.register_buffer("combo_init", combo_init)

        # 2. Set the offset and scale of the tanh map for m_Delta.
        # m_Delta map: r_0 = m_Delta,0 / b_0, eta_0 = atanh(r_0), s = m_Delta,0 / (b_0 (1 - r_0^2)),
        # so beta = 0 gives m_Delta,0, and a small step in beta changes m_Delta by the same relative amount as
        # the exp map changes the other nine; all ten coordinates are then on the same scale for Adam
        bound = self._m_diff_bound(combo_init)
        r0 = combo_init[self.M_DIFF_IX] / bound
        eta0 = torch.atanh(r0)
        scale = combo_init[self.M_DIFF_IX] / (bound * (1 - r0 ** 2))
        self.register_buffer("coordinate_version", torch.as_tensor(
            self.COORDINATE_VERSION, dtype=torch.int64, device=combo_init.device))
        self.register_buffer("m_diff_eta0", eta0)
        self.register_buffer("m_diff_scale", scale)

    @property
    def beta(self) -> Tensor:
        """Training coordinates; stored as free_params in checkpoints."""
        return self._parameters['free_params']

    @staticmethod
    def combos_of(theta: Tensor, Lb) -> Tensor:
        """The ten parameter combinations that determine the baseline matrices, theta_base(theta).

        k_b,Sigma = k_b1 + k_b2, c_b,Sigma = c_b1 + c_b2, m_Sigma = m_1 + m_2 + m_b, m_Delta = m_1 - m_2,
        J_eff = J_b + J_h + (L_b^2/4)(m_1 + m_2); the matrices depend on theta only through these ten.

        Args:
            theta: (14,) parameters theta.
            Lb: actuator distance L_b [m].
        Returns:
            (10,) identifiable combinations, in COMBO_NAMES order.
        """
        (kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d) = theta
        Lb = torch.as_tensor(Lb, dtype=theta.dtype, device=theta.device)
        _, _, J_eff = gantry_ss.inertia_constants(m1, m2, mb, mh, Jb, Jh, Lb)
        return torch.stack([
            kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh,
            m1 + m2 + mb, m1 - m2,
            J_eff, d,
        ])

    @classmethod
    def parameters_from_combinations(cls, combo: Tensor, theta_init: Tensor, Lb) -> Tensor:
        """Reconstruct theta from the identifiable combinations.

        Keep the nominal m_b and J_b : J_h ratio; split joint stiffness and damping equally.
        The reconstruction is not unique, but gives the same baseline matrices.

        Args:
            combo: (10,) theta_base.
            theta_init: (14,) theta^0, the source of m_b and J_b : J_h.
            Lb: actuator distance L_b [m].
        Returns:
            (14,) reconstructed parameters, in PARAM_NAMES order.
        """
        kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d = combo
        Lb = torch.as_tensor(Lb, dtype=combo.dtype, device=combo.device)
        mb = theta_init[cls.MB_IX]
        Jb0, Jh0 = theta_init[cls.JB_IX], theta_init[cls.JH_IX]
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
        """Bound on |m_Delta|, with m_Delta = m_1 - m_2 the mass of the X1 carriage minus that of the X2 carriage.
        The bound keeps M(Y) positive definite for every payload position Y.

        b = rho (2 / L_b) sqrt(m_Sigma J_eff), rho = M_DIFF_RHO = 0.99. With the other combinations positive,
        |m_Delta| < b keeps m_Sigma J_eff > (L_b^2/4) m_Delta^2; rho keeps the bound below that limit.

        Args:
            combo: (..., 10) physical combinations; uses m_Sigma and J_eff.
        Returns:
            Bound b [kg] on |m_1 - m_2|, with the last dimension of combo removed.
        """
        k = 2 * self.M_DIFF_RHO / self.Lb.to(combo.dtype)
        return k * torch.sqrt(combo[..., self.M_TOTAL_IX] * combo[..., self.J_EFF_IX])

    def combinations_from_beta(self, beta: Tensor) -> Tensor:
        """Map training coordinates beta to the physical combinations theta_base.

        theta_base,j = theta^0_base,j exp(beta_j) for nine positive combinations;
        m_Delta = b tanh(eta_0 + s beta_mDelta), with b from _m_diff_bound.

        Args:
            beta: (..., 10) training coordinates; zero gives combo_init.
        Returns:
            physical combinations theta_base in the same order and shape as beta.
        """
        i = self.M_DIFF_IX
        c = self.combo_init * torch.exp(beta)
        m_diff = self._m_diff_bound(c) * torch.tanh(
            self.m_diff_eta0 + self.m_diff_scale * beta[..., i])
        return torch.cat((c[..., :i], m_diff.unsqueeze(-1), c[..., i + 1:]), dim=-1)

    def beta_from_combinations(self, combo: Tensor) -> Tensor:
        """Training coordinates beta of given physical combinations; evaluation uses this to set the true baseline.

        beta_j = log(theta_base,j / theta^0_base,j) for the nine positive combinations;
        beta_mDelta = (atanh(m_Delta / b) - eta_0) / s for the signed mass difference.

        Args:
            combo: (..., 10) physical combinations in COMBO_NAMES order;
                positive except m_Delta, with |m_Delta| < _m_diff_bound(combo).
        Returns:
            dimensionless training coordinates beta in the same order and shape as combo; zero at combo_init.
        """
        i = self.M_DIFF_IX
        c = torch.as_tensor(combo, dtype=self.combo_init.dtype, device=self.combo_init.device)
        positive = torch.cat((c[..., :i], c[..., i + 1:]), dim=-1)
        # r = m_Delta / b, |r| < 1 required
        r = c[..., i] / self._m_diff_bound(c)
        init = torch.cat((self.combo_init[:i], self.combo_init[i + 1:]))
        # beta_j = log(theta_base,j / theta^0_base,j), beta_mDelta = (atanh(r) - eta_0) / s
        log = torch.log(positive / init)
        xi = (torch.atanh(r) - self.m_diff_eta0) / self.m_diff_scale
        return torch.cat((log[..., :i], xi.unsqueeze(-1), log[..., i:]), dim=-1)

    def recover_combinations(self) -> Tensor:
        """Current parameter estimate theta_base(beta), for reporting.

        Returns:
            (10,) estimated parameter combinations, in physical units.
        """
        return self.combinations_from_beta(self.beta)

    def transition_from_beta(self, beta: Tensor, z: Tensor) -> Tensor:
        """Baseline transition at the supplied beta, without changing the stored beta.

        OBC differentiates this transition with respect to beta to build the sensitivity columns of J.

        Args:
            beta: (10,) beta.
            z: (B, 9, 1) (x_tilde^n, u^n).
        Returns:
            (B, 6, 1) next normalised physical state.
        """
        mats = self._mats_at(beta)
        return self._rk4(z[:, : self.nx, :], z[:, self.nx :, :], mats)

    def mats_and_tangent_from_beta(self, beta: Tensor, dbeta: Tensor):
        """Baseline matrices at beta and their directional derivatives in direction dbeta, computed with torch.func.jvp.

        OBC uses dbeta = a and passes both tuples to rk4_with_tangent to compute varphi(beta_lin) a.

        Args:
            beta: (10,) beta (beta_lin for the OBC correction).
            dbeta: (10,) direction in the training coordinates.
        Returns:
            the matrix tuple of _mats_from_parameters and its directional derivative, with the same layout.
        """
        return torch.func.jvp(self._mats_at, (beta,), (dbeta,))

    def _mats_at(self, beta: Tensor):
        """Matrices at beta: beta -> theta_base -> theta (parameters_from_combinations) -> matrices."""
        return self._mats_from_parameters(
            self.parameters_from_combinations(self.combinations_from_beta(beta), self.params_init, self.Lb))

    def estimation_parameters(self):
        """Names, start values and current values of the identifiable combinations."""
        return self.COMBO_NAMES, self.combo_init.detach(), self.recover_combinations().detach()
