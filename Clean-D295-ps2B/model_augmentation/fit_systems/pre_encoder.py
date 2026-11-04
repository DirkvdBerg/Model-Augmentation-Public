"""Estimate each window's initial physical and added states from its past normalised inputs and outputs."""
# CHANGED: replaced Jan's linear_encoder_init with AugmentedEncoder; least_squares_start fits the physical rows
# to simulated baseline states instead of computing them from A, B, C, D, which hold only near one Y (LPV baseline).
# CHANGED: removed unused state-data helpers, SS_pre_encoder and the example; original: Model-Augmentation-Public-main
from torch import nn
import torch
import numpy as np

from model_augmentation.utils.utils import added
from model_augmentation.utils.torch_nets import build_mlp


@added
class AugmentedEncoder(nn.Module):
    """Estimate the initial physical states x_b (x_tilde) and added states x_a (x_bar) from the past normalised
    inputs and outputs:

    x_b = Wb_u u_hist + Wb_y y_hist + psi_tilde_b(chi)
    x_a = Wa_u u_hist + Wa_y y_hist + psi_tilde_a(chi)

    chi contains the input history followed by the output history, each written as one vector.
    psi_tilde_b and psi_tilde_a are the physical and added rows of the network correction; psi = (x_b, x_a).
    Extends Jan's linear-map-plus-network encoder with added states and a least-squares physical-state start.
    """

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
        """Initialize W^b, W^a and the tanh correction psi_tilde.

        Added states start as W^a chi. Before training, least_squares_start sets the physical estimate to W^b chi + b
        by overwriting W^b and psi_tilde's physical output rows.

        Args:
            nx: number of physical states (6).
            nu, ny: input and output channels (3, 3).
            na, nb: output and input history lengths minus one; model.py subtracts one from the model's lengths.
            nx_aug: number of added states.
            n_nodes_per_layer, n_hidden_layers: width and number of hidden layers of psi_tilde.
            dtype: dtype of W^b and W^a; model.py casts the whole encoder to this dtype.
        """
        super().__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.nx_aug = nx_aug
        self.na = na
        self.nb = nb

        # 1. Physical-state map W^b; n_y and n_u count all values in the flattened histories.
        self.n_y, self.n_u = (na + 1) * ny, (nb + 1) * nu
        # U(-1/sqrt(m), 1/sqrt(m)), m = n_y + n_u, as for nn.Linear with m inputs.
        # Keep these draws despite the later fit: removing them changes subsequent seeded initialisation.
        bound = 1.0 / np.sqrt(self.n_y + self.n_u)
        self.Wb_psi_y = nn.Parameter(torch.empty(nx, self.n_y, dtype=dtype).uniform_(-bound, bound))
        self.Wb_psi_u = nn.Parameter(torch.empty(nx, self.n_u, dtype=dtype).uniform_(-bound, bound))

        # 2. Added-state map W^a: each row starts as a random linear combination of the history.
        # Xavier uniform, gain 1: U(-a, a), a = sqrt(6 / (nx_aug + n_y + n_u)); draw both histories together.
        wa = torch.empty(nx_aug, self.n_y + self.n_u, dtype=dtype)
        nn.init.xavier_uniform_(wa, gain=1.0)
        self.Wa_psi_y = nn.Parameter(wa[:, :self.n_y].clone())
        self.Wa_psi_u = nn.Parameter(wa[:, self.n_y:].clone())

        # 3. Nonlinear correction to all states; build_mlp uses the default dtype, cast later by model.py.
        self.net, final_layer = build_mlp(self.n_u + self.n_y, nx + nx_aug, n_nodes_per_layer, n_hidden_layers,
                                          nn.Tanh)
        # build_mlp zeros all output biases; zeroing these weights makes the added-state correction zero.
        with torch.no_grad():
            final_layer.weight[nx:].zero_()

    def forward(self, uhist, yhist):
        """Estimate x_hat^n_{tau|tau} from the history ending at tau - 1; sample tau is excluded.

        Args:
            uhist: (B, nb + 1, nu) past normalised inputs, oldest first.
            yhist: (B, na + 1, ny) past normalised outputs, oldest first.
        Returns:
            (B, nx + nx_aug) initial state: normalised physical states, then added states.
        """
        # 1. Flatten time and channels; apply W^b and W^a to the two histories separately.
        uhist_mod = uhist.view(uhist.size(0), self.n_u, 1)
        yhist_mod = yhist.view(yhist.size(0), self.n_y, 1)

        x_b = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod
        x_a = self.Wa_psi_u @ uhist_mod + self.Wa_psi_y @ yhist_mod

        x = torch.cat([x_b, x_a], dim=1)

        x = x.view(-1, self.nx + self.nx_aug)

        # 2. psi_tilde receives input history first, then output history, as in Jan's encoder.
        return x + self.net(
            torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1)
        )
