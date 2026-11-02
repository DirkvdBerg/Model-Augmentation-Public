"""Jan Hoekstra's black-box nets, encoder psi and state map f_bb (deepSI 0.3.29 nets, 2 tanh hidden layers).
`.reshape` replaces deepSI's `.view`, same values; module and class names are fixed by the pickled thesis checkpoints.
Batch rows are windows; u = (r, u^ms) and e = y - r."""
__project_origin__ = "added"

import torch
from deepSI.fit_systems.encoders import default_encoder_net, default_state_net


class JanEncoder(default_encoder_net):
    def forward(self, upast, ypast):
        """Encoder of the black box.

        x^bb_0 = psi(u_past, e_past).
        Gives the state at the window start from the past n + 1 samples, so a simulation needs no initial state.

        Args:
            upast: (B, n_b, n_u) past inputs u^n = (r, u^ms), normalised.
            ypast: (B, n_a, n_y) past servo errors e^n, normalised.
        Returns:
            (B, n_x) start states x^bb_0.
        """
        return self.net(torch.cat([upast.reshape(upast.shape[0], -1),
                                   ypast.reshape(ypast.shape[0], -1)], dim=1))


class JanStateNet(default_state_net):
    def forward(self, x, u):
        """State map of the black box.

        x^bb_k+1 = f_bb(x^bb_k, u_k).

        Args:
            x: (B, n_x) states x^bb_k.
            u: (B, n_u) inputs u^n_k = (r_k, u^ms_k), normalised.
        Returns:
            (B, n_x) next states x^bb_k+1.
        """
        return self.net(torch.cat([x, u.reshape(u.shape[0], -1)], dim=1))
