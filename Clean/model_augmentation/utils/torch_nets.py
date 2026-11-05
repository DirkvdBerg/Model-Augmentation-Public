"""Shared layer construction for the augmentation network phi_aug and encoder correction psi_tilde."""
# CHANGED: Jan's zero_init_feed_forward_nn layers, but only the output bias is zeroed; callers zero chosen rows.
# CHANGED: removed unused network classes; original: Model-Augmentation-Public-main
from torch import nn
import torch

from model_augmentation.utils.utils import added


@added
def build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation):
    """Build hidden Linear/activation layers followed by a linear output layer.

    h_0 = z, h_l = activation(W_l h_{l-1} + b_l), output = W_L h_{L-1} + b_L.
    All weights and hidden biases retain nn.Linear defaults; only b_L is zeroed.

    Args:
        n_in, n_out: input and output size.
        n_nodes_per_layer: width n_h of each hidden layer.
        n_hidden_layers: number of hidden layers; 0 also gives one hidden layer.
        activation: hidden-layer activation class; both callers use nn.Tanh.
    Returns:
        network (nn.Sequential) and its output layer, so callers can initialize selected output rows.
    """
    # Keep layer creation order: changing it changes which seeded random draws initialize each layer.
    # CHANGED: no check that n_hidden_layers > 0; 0 also gives one hidden layer
    seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
    for _ in range(n_hidden_layers - 1):
        seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
        seq.append(activation())

    final_layer = nn.Linear(n_nodes_per_layer, n_out)
    seq.append(final_layer)

    net = nn.Sequential(*seq)

    # Zero bias lets callers zero selected outputs by changing only their output-layer weights.
    nn.init.constant_(final_layer.bias, val=0.0)
    return net, final_layer


@added
class AugmentationNetwork(nn.Module):
    """Augmentation network phi_aug(x_hat^n, u^n): physical-state corrections f_aug^n, then added states g_aug^n."""

    def __init__(self, n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation, nx_phys):
        """Zero the physical correction so the physical transition starts as the baseline.

        Unlike Jan's fully zero output layer, the added-state weights remain random to give g_aug^n a nonzero map.

        Args:
            n_in: input size 6 + n_xbar + 3.
            n_out: output size 6 + n_xbar.
            n_nodes_per_layer, n_hidden_layers, activation: as in build_mlp.
            nx_phys: number of physical output rows (6).
        """
        super().__init__()
        self.net, final_layer = build_mlp(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        with torch.no_grad():
            # build_mlp already zeros the bias; zero only the f_aug^n weights, keeping g_aug^n random.
            final_layer.weight[:nx_phys].zero_()

    def forward(self, X):
        """Compute the physical-state corrections and next added states.

        Args:
            X: (B, 6 + n_xbar + 3) normalised physical state, added states and normalised input.
        Returns:
            (B, 6 + n_xbar) normalised physical-state corrections, then next added states.
        """
        return self.net(X)
