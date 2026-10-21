from torch import nn
import torch
import numpy as np

from model_augmentation.utils.utils import added
from model_augmentation.utils.torch_nets import mlp_stack


@added
class linear_encoder_init_aug(nn.Module):
    """Extension of Jan's linear_encoder_init to augmented states, implementing
    Hoekstra 2026 Eq. 8 for the dynamic augmentation case (nx_aug > 0)."""

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
        # CHANGED: dtype of the constructed reconstructability map.
        dtype,
        activation=nn.Tanh,
    ):
        super(linear_encoder_init_aug, self).__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nx_aug = nx_aug
        self.na = na
        self.nb = nb

        n = na
        # THEORY: nn.Linear's default U(-1/sqrt(m), 1/sqrt(m)), m = fan-in = (n+1)(nu+ny): a
        # randomly initialised encoder, as in the nonlinear-baseline method of the encoder paper
        # (Eq. 35, D-251). The physical rows are then fitted to data before training (D-261).
        bound = 1.0 / np.sqrt((n + 1) * (nu + ny))
        self.Wb_psi_y = nn.Parameter(torch.empty(nx, (n + 1) * ny, dtype=dtype).uniform_(-bound, bound))
        self.Wb_psi_u = nn.Parameter(torch.empty(nx, (n + 1) * nu, dtype=dtype).uniform_(-bound, bound))

        _ny_hist = (n + 1) * ny
        _nu_hist = (n + 1) * nu
        # THEORY: Hoekstra et al. (2026), Eq. (31), prescribes Xavier for psi_aug.
        # HEURISTIC: uniform, gain 1, and one logical [W_y W_u] draw are the concrete
        # repository choices because the paper does not specify those details.
        wa = torch.empty(nx_aug, _ny_hist + _nu_hist, dtype=dtype)
        nn.init.xavier_uniform_(wa, gain=1.0)
        self.Wa_psi_y = nn.Parameter(wa[:, :_ny_hist].clone())
        self.Wa_psi_u = nn.Parameter(wa[:, _ny_hist:].clone())

        n_in, n_out = nu * (nb + 1) + ny * (na + 1), nx + nx_aug
        self.net, final_layer = mlp_stack(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        # CHANGED (D-261, thesis eq:aug_init): the physical rows keep the nn.Linear draw for
        # the data-based pre-fit; the added-state rows start at zero.
        with torch.no_grad():
            final_layer.weight[nx:].zero_()

    def forward(self, uhist, yhist):
        uhist_mod = uhist.view(uhist.size(0), self.nu * (self.nb + 1), 1)
        yhist_mod = yhist.view(yhist.size(0), self.ny * (self.na + 1), 1)

        x_b = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod
        x_a = self.Wa_psi_u @ uhist_mod + self.Wa_psi_y @ yhist_mod

        x = torch.cat([x_b, x_a], dim=1)

        x = x.view(-1, self.nx + self.nx_aug)

        return x + self.net(
            torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1)
        )
