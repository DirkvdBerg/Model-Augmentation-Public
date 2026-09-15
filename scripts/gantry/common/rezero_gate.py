"""ReZero-style gate for the augmentation ANN: zero OUTPUT without a zero output PROJECTION.

WHY. The augmented states (rows 6-7) reach the loss through one path only: they are the ANN's
input. `zero_init_feed_forward_nn` zeroes the FINAL LAYER's weight and bias
(`torch_nets.py:113-114`), so the ANN's input-Jacobian is exactly zero and the encoder's augmented
rows `W^a` receive exactly 0.000000e+00 gradient at initialisation (measured, gate G1 in
`verify_ms_gradient.py`; D-130).

THE DISTINCTION THIS MODULE IMPLEMENTS. ReZero (Bachlechner et al., UAI 2021, arXiv:2003.04887)
and Fixup (Zhang et al., ICLR 2019, arXiv:1901.09321) also start a branch at exactly zero output,
but they do it with a zero SCALAR GATE over a NORMALLY-INITIALISED branch, not with a zero output
projection. Both give the same function at initialisation; they differ in what gets gradient:

    ours today      w = F_0(z),  F_0 final layer == 0     ->  dw/dz == 0  AND  F(z) == 0
    gated (here)    w = alpha * F(z),  alpha == 0         ->  dw/dz == 0  BUT  F(z) != 0

The second column is the point. `dL/d alpha = <dL/dw, F(z)>`, so the gate receives a generically
NON-ZERO gradient at step 1 precisely because the branch is live, and once `alpha != 0` the
input-Jacobian `alpha * F'(z)` is non-zero too. arXiv:2607.16568 (2026) states this as an
escape-from-initialization proposition and names the combination of a zero output projection WITH a
zero gate as an exact saddle that gradient descent cannot escape. That paper is an unrefereed
preprint and was read at abstract level only; the algebra above does not depend on it.

WHAT THIS DOES NOT CLAIM. The gate does not make `W^a` gradient non-zero AT INITIALISATION. It
cannot: with no identity path from z to w, `dw/dz = alpha F'(z) = 0` while `alpha = 0`. The claim
is about step 2 onward. Whether the plain zero-init ALSO recovers by step 2 (its final layer does
receive gradient at step 1) is an empirical question, and it is the control arm in
`scripts/gantry/coulomb-offset/verify_rezero_gate.py`. Do not assume this module is an improvement
until that comparison has been read.

IMPLEMENTATION. `torch.nn.utils.parametrize` on the final Linear's weight AND bias, sharing one
scalar `alpha`, so the module structure is unchanged: `.net` stays an `nn.Sequential`, state_dicts
keep their shapes, and `apply_lipschitz_cap` still composes. The bias is gated as well as the
weight: an ungated bias would start at zero but receive gradient immediately, which is exactly the
free DC term this project has been fighting (D-090 run table, the dY DC).

Enable with `ANN_REZERO_GATE=1` (see `model.py`). Default off, so nothing changes unless asked.
"""
__project_origin__ = "added"

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization


class _Gate(nn.Module):
    """Multiplies a tensor by a shared scalar. Owns alpha only if `alpha` is None."""

    def __init__(self, alpha: nn.Parameter = None):
        super().__init__()
        if alpha is None:
            # THEORY: ReZero (Bachlechner et al. 2021) gates each branch with a single
            # zero-initialised scalar parameter.
            self.alpha = nn.Parameter(torch.zeros(()))
        else:
            # Hide the shared Parameter from this module's registry (a list is not traversed by
            # nn.Module), so it is owned by exactly one parametrization and appears once in
            # .parameters(). Registering it twice would double its gradient.
            self._shared = [alpha]

    @property
    def gate(self) -> torch.Tensor:
        return self.alpha if hasattr(self, 'alpha') else self._shared[0]

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return self.gate * W


def apply_rezero_gate(net_module: nn.Module, std: float = None) -> nn.Parameter:
    """Re-initialise the final Linear of an MLP normally, then gate it with a zero scalar.

    `net_module` is the `zero_init_feed_forward_nn` (it carries `.net`, an `nn.Sequential` whose
    last entry is the final `nn.Linear`). Returns the shared `alpha` Parameter so callers can
    report or monitor it.

    Output after this call is still EXACTLY zero, which the caller should assert rather than trust.
    """
    seq = net_module.net
    final = None
    for m in reversed(list(seq)):
        if isinstance(m, nn.Linear):
            final = m
            break
    if final is None:
        raise ValueError('no nn.Linear found in the ANN; cannot apply the ReZero gate')

    # Undo the zero-init of torch_nets.py:113-114. Without this the branch output F(z) is
    # identically zero, alpha receives no gradient either, and the configuration is the exact
    # saddle rather than the fix.
    # THEORY: nn.Linear's own default (Kaiming-uniform on weight, fan-in uniform on bias) is what
    # every non-final layer in this MLP already carries; reset_parameters restores exactly that.
    final.reset_parameters()
    if std is not None:
        with torch.no_grad():
            nn.init.normal_(final.weight, mean=0.0, std=float(std))
            nn.init.zeros_(final.bias)

    gate_w = _Gate()
    register_parametrization(final, 'weight', gate_w)
    register_parametrization(final, 'bias', _Gate(gate_w.alpha))
    return gate_w.alpha
