from torch import nn
import torch

from model_augmentation.utils.utils import added


@added
def mlp_stack(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation):
    """`[Linear, act, (Linear, act) x (n_hidden_layers - 1), Linear]` with a zero final bias, as
    (Sequential, final layer). The layers draw their default initialisation in this order."""
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
class live_g_feed_forward_nn(nn.Module):
    """zero_init_feed_forward_nn, except the added-state rows of the final weight stay live."""

    def __init__(self, n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation, nx_phys):
        super(live_g_feed_forward_nn, self).__init__()
        self.net, final_layer = mlp_stack(n_in, n_out, n_nodes_per_layer, n_hidden_layers, activation)
        # the routed outputs are the states in order, so the physical rows are the first nx_phys
        with torch.no_grad():
            final_layer.weight[:nx_phys].zero_()

    def forward(self, X):
        return self.net(X)
