from torch import nn
import torch

from model_augmentation.utils.utils import added


@added
class live_g_feed_forward_nn(nn.Module):
    """zero_init_feed_forward_nn, except the added-state rows of the final weight stay live."""

    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64,
                 n_hidden_layers=2, activation=nn.Tanh,
                 output_state_indices=(), nx_phys=0):
        super(live_g_feed_forward_nn, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in, n_nodes_per_layer), activation()]
        for i in range(n_hidden_layers - 1):
            seq.append(nn.Linear(n_nodes_per_layer, n_nodes_per_layer))
            seq.append(activation())

        final_layer = nn.Linear(n_nodes_per_layer, n_out)
        seq.append(final_layer)

        self.net = nn.Sequential(*seq)

        nn.init.constant_(final_layer.bias, val=0.0)
        with torch.no_grad():
            for output_row, state_index in enumerate(output_state_indices):
                if int(state_index) < int(nx_phys):
                    final_layer.weight[output_row].zero_()

    def forward(self, X):
        return self.net(X)
