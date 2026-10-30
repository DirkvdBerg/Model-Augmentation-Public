"""The encoder of the augmented model.

Conventions: superscript n = normalised; chi = (y history, u history), samples as rows; psi_tilde reads the fixed
permutation (u history, y history). Code names: Wb_psi_* = W^b, Wa_psi_* = W^a, net = psi_tilde, nx_aug = n_xbar.
"""
# CHANGED: restricted to the thesis training path; full implementation in
# model_augmentation/fit_systems/pre_encoder.py at the repository root
from torch import nn
import torch
import numpy as np

# CHANGED: the @added marker and the layer stack of the added encoder
from model_augmentation.utils.utils import added
from model_augmentation.utils.torch_nets import build_mlp


@added
class AugmentedEncoder(nn.Module):
    """Encoder x_hat^n_{tau|tau} = psi(chi_tau) = [W^b; W^a] chi_tau + psi_tilde(chi_tau), chi_tau = the n + 1 past
    y^n and u^n (SUBNET, Beintema et al.; Hoekstra et al. Eq. (8))."""

    def __init__(
        self,
        nx,
        nu,
        ny,
        na,
        nb,
        nx_aug,
        n_nodes_per_layer,
        n_hidden_layers,
        dtype,
    ):
        """Encoder psi = [W^b; W^a] chi + psi_tilde(chi).

        W^b ~ U(-1/sqrt m, 1/sqrt m) (set later by the encoder start), W^a Xavier, psi_tilde's added-state rows at zero.

        Args:
            nx: 6 physical states; nx_aug: n_xbar.
            nu, ny: input and output channels (3, 3).
            na, nb: history lengths minus one (the window holds na + 1 and nb + 1 samples).
            n_nodes_per_layer, n_hidden_layers: psi_tilde's width n_h and depth.
            dtype: model dtype.
        """
        super().__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nx_aug = nx_aug
        self.na = na
        self.nb = nb

        # 1. Physical linear rows W^b.
        # the windows hold na + 1 output and nb + 1 input samples
        self.n_y, self.n_u = (na + 1) * ny, (nb + 1) * nu
        # THEORY: W^b ~ U(-1/sqrt(m), 1/sqrt(m)), m = (n + 1)(n_y + n_u) (code n_y + n_u, the flattened window
        # widths), the nn.Linear default; drawn although
        # prefit_physical_encoder overwrites it, so the later draws stay the same
        bound = 1.0 / np.sqrt(self.n_y + self.n_u)
        self.Wb_psi_y = nn.Parameter(torch.empty(nx, self.n_y, dtype=dtype).uniform_(-bound, bound))
        self.Wb_psi_u = nn.Parameter(torch.empty(nx, self.n_u, dtype=dtype).uniform_(-bound, bound))

        # 2. Added linear rows W^a (Xavier).
        # THEORY: W^a ~ U(-a, a), a = sqrt(6 / (n_xbar + m)), Xavier uniform as Hoekstra et al. prescribe
        # for psi_a
        # HEURISTIC: gain 1, and one draw for [W^a_y W^a_u] together; the paper does not fix these details
        wa = torch.empty(nx_aug, self.n_y + self.n_u, dtype=dtype)
        nn.init.xavier_uniform_(wa, gain=1.0)
        self.Wa_psi_y = nn.Parameter(wa[:, :self.n_y].clone())
        self.Wa_psi_u = nn.Parameter(wa[:, self.n_y:].clone())

        # 3. Nonlinear correction psi_tilde.
        self.net, final_layer = build_mlp(self.n_u + self.n_y, nx + nx_aug, n_nodes_per_layer, n_hidden_layers,
                                          nn.Tanh)
        # Added-state output rows of psi_tilde: W~^xbar_L = 0 (b~_L = 0 from build_mlp); the physical rows keep
        # their draw until prefit_physical_encoder sets them
        with torch.no_grad():
            final_layer.weight[nx:].zero_()

    def forward(self, uhist, yhist):
        """Encoder psi.

        x_hat^n_{tau|tau} = [W^b; W^a] chi_tau + psi_tilde(chi_tau), chi_tau = (y history, u history) up to tau - 1.

        Args:
            uhist: (B, n + 1, 3) past u^n; yhist: (B, n + 1, 3) past y^n, normalised.
        Returns:
            (B, 6 + n_xbar) x_hat^n_{tau|tau}: physical states then added states.
        """
        uhist_mod = uhist.view(uhist.size(0), self.n_u, 1)
        yhist_mod = yhist.view(yhist.size(0), self.n_y, 1)

        # Linear part: (W^b chi, W^a chi).
        x_b = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod
        x_a = self.Wa_psi_u @ uhist_mod + self.Wa_psi_y @ yhist_mod

        x = torch.cat([x_b, x_a], dim=1)

        x = x.view(-1, self.nx + self.nx_aug)

        # Nonlinear correction: + psi_tilde, which reads (u history, y history), a fixed permutation of chi.
        return x + self.net(
            torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1)
        )
