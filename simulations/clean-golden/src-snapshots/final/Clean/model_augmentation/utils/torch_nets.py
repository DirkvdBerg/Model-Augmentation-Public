"""The layer stack of the networks and the augmentation network phi_aug.

Conventions: superscript n = normalised; the network input is (x_tilde^n, xbar, u^n), its outputs are routed to
the states in order (6 physical rows, then n_xbar added rows).
"""
# CHANGED: restricted to the thesis training path; full implementation in
# model_augmentation/utils/torch_nets.py at the repository root
from torch import nn
import torch

from model_augmentation.utils.utils import added  # CHANGED: the marker of the added definitions


@added
def build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation):
    """Multilayer perceptron of phi_aug, psi_tilde and the decoder.

    phi(z) = W_L h + b_L, h_l = tanh(W_l h_{l-1} + b_l); hidden layers from U(-1/sqrt m, 1/sqrt m) (nn.Linear
    default), b_L = 0.

    Args:
        n_in, n_out: input and output size; n_nodes_per_layer: n_h; n_hidden_layers: at least 1; activation: tanh.
    Returns:
        (Sequential, output layer).
    """
    # keep constructor draws in this order to preserve seeded initialisation
    seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
    for _ in range(n_hidden_layers - 1):
        seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
        seq.append(activation())

    final_layer = nn.Linear(n_nodes_per_layer, n_out)
    seq.append(final_layer)

    net = nn.Sequential(*seq)

    # b_L = 0.
    nn.init.constant_(final_layer.bias, val=0.0)
    return net, final_layer


@added
class AugmentationNetwork(nn.Module):
    """phi_aug: (f_aug^n, g_aug^n) = MLP(x_hat^n, u^n), with W^xtilde_L = 0 at the start (so f_aug^n = 0) and
    W^xbar_L ~ U(-1/sqrt n_h, 1/sqrt n_h)."""

    def __init__(self, n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation, nx_phys):
        """Augmentation network phi_aug at its start.

        An MLP (build_mlp) whose first nx_phys output rows are f_aug^n, set to zero, and the rest g_aug^n, kept random.

        Args:
            n_in: 6 + n_xbar + 3; n_out: 6 + n_xbar.
            n_nodes_per_layer, n_hidden_layers, activation: n_h, depth, tanh.
            nx_phys: number of physical output rows (6).
        """
        super().__init__()
        self.net, final_layer = build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        # the routed outputs are the states in order, so the physical rows are the first nx_phys
        with torch.no_grad():
            # W^xtilde_L = 0, so f_aug^n = 0 at the start.
            final_layer.weight[:nx_phys].zero_()

    def forward(self, X):
        """Augmentation network phi_aug.

        (f_aug^n, g_aug^n) = phi_aug(x_hat^n, u^n).

        Args:
            X: (B, 6 + n_xbar + 3) (x_tilde^n, xbar, u^n).
        Returns:
            (B, 6 + n_xbar): physical corrections, then added-state updates.
        """
        return self.net(X)
