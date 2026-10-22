"""The encoder of the augmented model."""
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
    """SUBNET encoder (Beintema et al.) of the physical and added states: psi(chi) = [W^b; W^a] chi + psi_tilde(chi)."""

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
        activation=nn.Tanh,
    ):
        super(AugmentedEncoder, self).__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nx_aug = nx_aug
        self.na = na
        self.nb = nb

        # the windows hold n + 1 samples
        n = na
        # THEORY: W^b ~ U(-1/sqrt(m), 1/sqrt(m)), m = (n+1)(nu+ny), the nn.Linear default; drawn although
        # prefit_physical_encoder overwrites it, so the later draws stay the same
        bound = 1.0 / np.sqrt((n + 1) * (nu + ny))
        self.Wb_psi_y = nn.Parameter(torch.empty(nx, (n + 1) * ny, dtype=dtype).uniform_(-bound, bound))
        self.Wb_psi_u = nn.Parameter(torch.empty(nx, (n + 1) * nu, dtype=dtype).uniform_(-bound, bound))

        n_output_history_values = (n + 1) * ny
        n_input_history_values = (n + 1) * nu
        # THEORY: W^a ~ U(-a, a), a = sqrt(6 / (n_a + (n+1)(nu+ny))), Xavier uniform as Hoekstra et al. prescribe
        # for psi_a
        # HEURISTIC: gain 1, and one draw for [W^a_y W^a_u] together; the paper does not fix these details
        wa = torch.empty(nx_aug, n_output_history_values + n_input_history_values, dtype=dtype)
        nn.init.xavier_uniform_(wa, gain=1.0)
        self.Wa_psi_y = nn.Parameter(wa[:, :n_output_history_values].clone())
        self.Wa_psi_u = nn.Parameter(wa[:, n_output_history_values:].clone())

        n_in, n_out = nu * (nb + 1) + ny * (na + 1), nx + nx_aug
        self.net, final_layer = build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        # the added-state output rows of psi_tilde start at zero; its physical rows keep their draw until
        # prefit_physical_encoder sets them
        with torch.no_grad():
            final_layer.weight[nx:].zero_()

    def forward(self, uhist, yhist):
        """Estimate physical and added initial states from input and output histories."""
        uhist_mod = uhist.view(uhist.size(0), self.nu * (self.nb + 1), 1)
        yhist_mod = yhist.view(yhist.size(0), self.ny * (self.na + 1), 1)

        # [W^b; W^a] chi, chi = (y history, u history)
        x_b = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod
        x_a = self.Wa_psi_u @ uhist_mod + self.Wa_psi_y @ yhist_mod

        x = torch.cat([x_b, x_a], dim=1)

        x = x.view(-1, self.nx + self.nx_aug)

        # psi_tilde reads (u history, y history), a fixed permutation of chi
        return x + self.net(
            torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1)
        )
