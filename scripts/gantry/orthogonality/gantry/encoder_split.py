"""Output-preserving separation of the encoder's physical and latent nonlinear correction.

THE PROBLEM, stated exactly. `linear_encoder_init_aug` (pre_encoder.py) computes

    x_0 = [ W^b_u u_h + W^b_y y_h - x_off ; W^a_u u_h + W^a_y y_h ]  +  net([u_h; y_h])

with ONE `net` whose `nx + nx_aug` outputs are split across the physical and the latent rows.
The specification's ownership (Eqs. (7)-(8)) puts the physical initialisation in the nuisance
group `xi_p` and the latent initialisation in the augmentation group `eta_a`. Those two groups
must be DISJOINT BY TENSOR IDENTITY: every hidden weight of `net` currently feeds both, so
slicing the output, or detaching one slice, changes nothing about which parameters the two
depend on. The specification says this in as many words: "Slicing outputs or detaching one does
not make its trainable dependencies disjoint."

THE MIGRATION, spec Sect. 4.2 default. Keep `Wb_psi_*` and `Wa_psi_*`. Replace `net` with two
independent branches `net_p` and `net_a`: the hidden layers are DEEP COPIES of the shared ones,
and the final layer of each keeps only the rows it owns (`0:nx` and `nx:nx+nx_aug`) with the
matching bias entries. For a feedforward stack this reproduces the original outputs exactly,
because the hidden activations are identical and a Linear's output rows are independent.

WHAT IT DOES NOT DO. It changes parameter SHARING, so the converted model and the original are
the same function but not the same optimisation problem. Every arm of a comparison must use the
converted encoder, optimiser state must be reset identically, and an unregularised
converted-versus-original control belongs in the pilot, not here.
"""
__project_origin__ = "added"

import copy

import torch
from torch import nn

from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug


class SplitEncoderInitAug(linear_encoder_init_aug):
    """`linear_encoder_init_aug` with two independent nonlinear correction branches.

    Constructed only by `split_encoder`; the __init__ of the parent is deliberately NOT re-run,
    because rebuilding the reconstructability map would give a different (freshly initialised)
    module rather than a conversion of the one that was trained.
    """

    def forward(self, uhist, yhist):
        uhist_mod = uhist.view(uhist.size(0), self.nu * (self.nb + 1), 1)
        yhist_mod = yhist.view(yhist.size(0), self.ny * (self.na + 1), 1)
        if self.fix_enabled:
            uhist_mod = uhist_mod + self.u_off
            yhist_mod = yhist_mod + self.y_off
        x_b = self.Wb_psi_u @ uhist_mod + self.Wb_psi_y @ yhist_mod
        x_a = self.Wa_psi_u @ uhist_mod + self.Wa_psi_y @ yhist_mod
        if self.fix_enabled:
            x_b = x_b - self.x_off
        x_b = x_b.view(-1, self.nx)
        x_a = x_a.view(-1, self.nx_aug)
        if not self.flag_linear_only:
            z = torch.cat((uhist.view(uhist.size(0), -1), yhist.view(yhist.size(0), -1)), dim=1)
            x_b = x_b + self.net_p(z)
            x_a = x_a + self.net_a(z)
        return torch.cat([x_b, x_a], dim=1)

    # ---- the two ownership views the adapter needs -------------------------------------
    def physical_parameters(self):
        """xi_p: everything that can move the PHYSICAL initial state and nothing else."""
        ps = [self.Wb_psi_u, self.Wb_psi_y]
        if not self.flag_linear_only:
            ps += list(self.net_p.parameters())
        return ps

    def latent_parameters(self):
        """eta_a: everything that can move the LATENT initial state and nothing else."""
        ps = [self.Wa_psi_u, self.Wa_psi_y]
        if not self.flag_linear_only:
            ps += list(self.net_a.parameters())
        return ps

    def encode_split(self, uhist, yhist):
        """(x_phys_0, x_lat_0) separately, so a caller can hold one and differentiate the other."""
        x = self.forward(uhist, yhist)
        return x[:, :self.nx], x[:, self.nx:]


def _final_linear(seq):
    ix = [i for i, m in enumerate(seq) if isinstance(m, nn.Linear)]
    return ix[-1]


def split_encoder(enc: linear_encoder_init_aug) -> SplitEncoderInitAug:
    """Return a converted COPY. The input module is not modified."""
    if not isinstance(enc, linear_encoder_init_aug):
        raise TypeError(
            f'the split migration is defined for linear_encoder_init_aug, got '
            f'{type(enc).__name__}. The specification requires implementing the conversion for '
            f'the selected encoder explicitly or rejecting the configuration; it must not be '
            f'guessed.')
    new = copy.deepcopy(enc)
    new.__class__ = SplitEncoderInitAug
    if getattr(enc, 'flag_linear_only', False):
        return new
    nx, nxa = enc.nx, enc.nx_aug
    seq = enc.net
    li = _final_linear(seq)
    fin = seq[li]
    if fin.out_features != nx + nxa:
        raise RuntimeError(
            f'the shared correction net emits {fin.out_features} rows but the state split is '
            f'{nx} + {nxa}. Conversion refused.')

    def branch(rows):
        b = copy.deepcopy(seq)
        old = b[li]
        # `device=` AS WELL AS `dtype=`. Carrying only the dtype builds this layer on the CPU
        # default and splices it into an otherwise-CUDA Sequential; the `copy_` below happily
        # crosses devices, so nothing complains until the forward pass dies with "Expected all
        # tensors to be on the same device" (job 82436).
        newlin = nn.Linear(old.in_features, len(rows), bias=old.bias is not None,
                           dtype=old.weight.dtype, device=old.weight.device)
        with torch.no_grad():
            newlin.weight.copy_(old.weight[rows])
            if old.bias is not None:
                newlin.bias.copy_(old.bias[rows])
        b[li] = newlin
        return b

    new.net_p = branch(list(range(nx)))
    new.net_a = branch(list(range(nx, nx + nxa)))
    del new.net
    return new


def check_split_parity(orig, new, uhist, yhist, tol=0.0):
    """Initialisation-output parity, and DISJOINTNESS BY TENSOR IDENTITY.

    Both halves are required. Equal outputs with shared tensors is exactly the state the
    migration exists to leave, and disjoint tensors with different outputs is a different model.
    """
    with torch.no_grad():
        a = orig(uhist, yhist)
        b = new(uhist, yhist)
    err = float((a - b).abs().max())
    rel = err / max(float(a.abs().max()), 1e-300)
    pid = {id(t) for t in new.physical_parameters()}
    lid = {id(t) for t in new.latent_parameters()}
    return dict(max_abs=err, max_rel=rel, disjoint=len(pid & lid) == 0,
                n_phys=len(pid), n_lat=len(lid),
                n_phys_scalars=sum(t.numel() for t in new.physical_parameters()),
                n_lat_scalars=sum(t.numel() for t in new.latent_parameters()))
