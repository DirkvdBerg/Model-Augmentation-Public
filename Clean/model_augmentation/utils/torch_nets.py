"""The layer stack of the networks and the augmentation network phi_aug.

Conventions: superscript n = normalised; the network input is (x_tilde^n, xbar, u^n), its outputs are routed to
the states in order (6 physical rows, then n_xbar added rows).
"""
# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public,
# commit 8200923 (model_augmentation/__init__.py)
from torch import nn
import torch

from model_augmentation.utils.utils import added  # CHANGED: the marker of the added definitions


@added
def build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation):
    """Multilayer perceptron of phi_aug and psi_tilde.

    phi(z) = W_L h + b_L, h_l = tanh(W_l h_{l-1} + b_l); default nn.Linear initialisation, then b_L = 0.

    Args:
        n_in, n_out: input and output size.
        n_nodes_per_layer: width n_h of each hidden layer.
        n_hidden_layers: number of hidden layers, at least 1.
        activation: activation class (tanh).
    Returns:
        (Sequential, output layer).
    """
    # keep the layer order: it fixes the seeded initialisation
    seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
    for _ in range(n_hidden_layers - 1):
        seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
        seq.append(activation())

    final_layer = nn.Linear(n_nodes_per_layer, n_out)
    seq.append(final_layer)

    net = nn.Sequential(*seq)

    # b_L = 0
    nn.init.constant_(final_layer.bias, val=0.0)
    return net, final_layer


@added
class AugmentationNetwork(nn.Module):
    """phi_aug: (f_aug^n, g_aug^n) = MLP(x_hat^n, u^n); f_aug^n starts at zero, g_aug^n random."""

    def __init__(self, n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation, nx_phys):
        """Augmentation network phi_aug at its start.

        The f_aug^n rows start at zero, so the model starts as the baseline; the g_aug^n rows stay random, so the
        added states move from the first step.

        Args:
            n_in: input size 6 + n_xbar + 3.
            n_out: output size 6 + n_xbar.
            n_nodes_per_layer, n_hidden_layers, activation: as in build_mlp.
            nx_phys: number of physical output rows (6).
        """
        super().__init__()
        self.net, final_layer = build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        with torch.no_grad():
            # W^xtilde_L = 0, so f_aug^n = 0 at the start
            final_layer.weight[:nx_phys].zero_()

    def forward(self, X):
        """Augmentation network phi_aug.

        (f_aug^n, g_aug^n) = phi_aug(x_hat^n, u^n).

        Args:
            X: (B, 6 + n_xbar + 3) (x_tilde^n, xbar, u^n).
        Returns:
            (B, 6 + n_xbar): f_aug^n, then g_aug^n, normalised.
        """
        return self.net(X)
