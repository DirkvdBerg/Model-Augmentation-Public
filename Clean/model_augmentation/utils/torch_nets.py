"""The layer stack of the networks and the augmentation network phi_aug."""
# CHANGED: restricted to the thesis training path; full implementation in
# model_augmentation/utils/torch_nets.py at the repository root
from torch import nn
import torch

from model_augmentation.utils.utils import added  # CHANGED: the marker of the added definitions


@added
def build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation):
    """Build max(1, n_hidden_layers) hidden layers and a final Linear with zero bias; return (Sequential, final layer)."""
    # keep constructor draws in this order to preserve seeded initialisation
    seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
    for i in range(n_hidden_layers - 1):
        seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
        seq.append(activation())

    final_layer = nn.Linear(n_nodes_per_layer, n_out)
    seq.append(final_layer)

    net = nn.Sequential(*seq)

    nn.init.constant_(final_layer.bias, val=0.0)
    return net, final_layer


@added
class AugmentationNetwork(nn.Module):
    """phi_aug = (f_aug^n, g_aug^n): an MLP whose physical output rows start at zero, so f_aug^n = 0 at the start,
    while the added-state rows keep their random draw."""

    def __init__(self, n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation, nx_phys):
        super(AugmentationNetwork, self).__init__()
        self.net, final_layer = build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        # the routed outputs are the states in order, so the physical rows are the first nx_phys
        with torch.no_grad():
            final_layer.weight[:nx_phys].zero_()

    def forward(self, X):
        """Return the physical correction and added-state update from X."""
        return self.net(X)
