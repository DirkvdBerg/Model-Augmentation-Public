"""The encoder of the augmented model: it estimates the initial state of every training window from the past inputs
and outputs before it."""
# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public,
# commit 8200923 (model_augmentation/__init__.py)
from torch import nn
import torch
import numpy as np

# CHANGED: the @added marker and the layer stack of the added encoder
from model_augmentation.utils.utils import added
from model_augmentation.utils.torch_nets import build_mlp


@added
class AugmentedEncoder(nn.Module):
    """Encoder psi: a linear map of the past window plus a network correction (SUBNET, Beintema et al.; Hoekstra et
    al. Eq. (8))."""

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
        """Encoder psi(chi) = [W^b; W^a] chi + psi_tilde(chi), chi = the n + 1 past normalised outputs and inputs.

        The added states start as W^a chi (psi_tilde's added-state rows are zero); W^b and the physical rows are set
        later by the encoder start (least_squares_start).

        Args:
            nx: number of physical states (6).
            nu, ny: input and output channels (3, 3).
            na, nb: history lengths minus one (the window holds na + 1 and nb + 1 samples).
            nx_aug: number of added states n_xbar.
            n_nodes_per_layer, n_hidden_layers: width n_h and depth of psi_tilde.
            dtype: model dtype.
        """
        super().__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nx_aug = nx_aug
        self.na = na
        self.nb = nb

        # 1. Linear map of the physical states, W^b (Wb_psi_y on the output history, Wb_psi_u on the input history).
        # window sizes: n_y output values, n_u input values
        self.n_y, self.n_u = (na + 1) * ny, (nb + 1) * nu
        # THEORY: W^b ~ U(-1/sqrt(m), 1/sqrt(m)), m = window width, the nn.Linear default
        # drawn even though the encoder start overwrites it, so the random draws after it stay the same
        bound = 1.0 / np.sqrt(self.n_y + self.n_u)
        self.Wb_psi_y = nn.Parameter(torch.empty(nx, self.n_y, dtype=dtype).uniform_(-bound, bound))
        self.Wb_psi_u = nn.Parameter(torch.empty(nx, self.n_u, dtype=dtype).uniform_(-bound, bound))

        # 2. Linear map of the added states, W^a (Wa_psi_y, Wa_psi_u), drawn at random, so each added state starts
        #    as a different linear mix of the window.
        # THEORY: W^a ~ U(-a, a), a = sqrt(6 / (n_xbar + m)), Xavier uniform (Hoekstra et al.)
        # HEURISTIC: gain 1, one draw for [W^a_y W^a_u] together
        wa = torch.empty(nx_aug, self.n_y + self.n_u, dtype=dtype)
        nn.init.xavier_uniform_(wa, gain=1.0)
        self.Wa_psi_y = nn.Parameter(wa[:, :self.n_y].clone())
        self.Wa_psi_u = nn.Parameter(wa[:, self.n_y:].clone())

        # 3. Network correction psi_tilde (net): tanh network from the window to all nx + nx_aug states.
        self.net, final_layer = build_mlp(self.n_u + self.n_y, nx + nx_aug, n_nodes_per_layer, n_hidden_layers,
                                          nn.Tanh)
        # its added-state output rows start at zero, so the added states start as the linear W^a chi
        with torch.no_grad():
            final_layer.weight[nx:].zero_()

    def forward(self, uhist, yhist):
        """Encoder psi.

        x_hat^n_{tau|tau} = [W^b; W^a] chi_tau + psi_tilde(chi_tau), chi_tau = (y history, u history) up to tau - 1.

        Args:
            uhist: (B, n + 1, 3) past u^n, oldest first.
            yhist: (B, n + 1, 3) past y^n, oldest first.
        Returns:
            (B, 6 + n_xbar) x_hat^n_{tau|tau}: x_tilde^n, then the added states.
        """
        # 1. Linear part of physical and added states: (W^b chi, W^a chi).
        uhist_mod = uhist.view(uhist.size(0), self.n_u, 1)
        yhist_mod = yhist.view(yhist.size(0), self.n_y, 1)

        x_b = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod
        x_a = self.Wa_psi_u @ uhist_mod + self.Wa_psi_y @ yhist_mod

        x = torch.cat([x_b, x_a], dim=1)

        x = x.view(-1, self.nx + self.nx_aug)

        # 2. Add the network correction psi_tilde, which reads the window in the order (u history, y history).
        return x + self.net(
            torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1)
        )
