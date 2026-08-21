"""By-construction Lipschitz cap for the augmentation ANN (D-118, stability-preserving route).

The static augmentation ANN (x[k+1] = f_physics(x) + ANN(x,u)) can destabilize the long free-run
(v5: it makes the K=0/z=1 axis diverge ~50x vs the physics baseline). This module caps the ANN's
Lipschitz constant BY CONSTRUCTION, so its Jacobian contribution to the augmented state-transition map
is bounded -> the augmented model cannot inject unbounded gain. This is the static-ANN analog of the
Gyorok LFR-contraction / Revay Lipschitz-bounded-network route: the cap L plays the role of the
contraction rate alpha (a magnitude bound, not sign -- see D-117 for the marginal-integrator caveat).

`SpectralCap` is a SOFT per-layer spectral-norm cap: W -> W * cap/max(sigma(W), cap). While sigma<=cap
it returns W UNCHANGED, so (a) the zero-init final layer stays exactly zero (augmentation starts as the
pure baseline) and (b) weights grow GRADUALLY up to the cap during training -- unlike plain spectral
normalization (torch spectral_norm), which forces sigma=1 and would make the final layer jump to full
strength after the first gradient step, destroying the zero-init training dynamics. Only when sigma
exceeds cap is the layer scaled down. sigma is estimated by warm-started power iteration.

Overall MLP Lipschitz <= product of per-layer spectral norms (tanh is 1-Lipschitz), so capping each of
the n_linear Linear layers at L**(1/n_linear) bounds the whole net's Lipschitz at ~L.
"""
__project_origin__ = "added"

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization


class SpectralCap(nn.Module):
    # THEORY: soft spectral-norm cap; sigma via power iteration (Miyato et al. 2018 spectral norm),
    # relaxed to a CAP (scale only if sigma>cap) to preserve zero-init and gradual growth.
    def __init__(self, weight: torch.Tensor, cap: float, n_power: int = 1):
        super().__init__()
        self.cap = float(cap)
        self.n_power = int(n_power)
        u = torch.randn(weight.shape[0])
        self.register_buffer('u', u / (u.norm() + 1e-12))

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        u = self.u
        # Re-seed if the buffer has collapsed: registering the parametrization on the ZERO-INIT weight
        # runs power iteration on W=0, which drives u->0 (a fixed point that never recovers once W
        # becomes nonzero). Redraw a random unit vector whenever u degenerates.
        if (not torch.isfinite(u).all()) or (u.norm() < 1e-8):
            u = torch.randn_like(u); u = u / (u.norm() + 1e-12)
        with torch.no_grad():                       # warm-started power iteration for sigma_max(W)
            for _ in range(self.n_power):
                v = torch.mv(W.t(), u); vn = v.norm()
                if vn < 1e-12:                       # W u ~ 0 (e.g. W==0): sigma=0 <= cap, no scaling
                    self.u.copy_(u)
                    return W
                v = v / vn
                u = torch.mv(W, v); u = u / (u.norm() + 1e-12)
            self.u.copy_(u)
        v = torch.mv(W.t(), u); v = v / (v.norm() + 1e-12)
        sigma = torch.dot(u, torch.mv(W, v))        # differentiable Rayleigh-quotient estimate
        scale = self.cap / torch.clamp(sigma, min=self.cap)   # =1 if sigma<=cap else cap/sigma (<1)
        return W * scale


def apply_lipschitz_cap(net_module: nn.Module, L: float, n_power: int = 1) -> int:
    """Cap the overall Lipschitz constant of an MLP at ~L by capping each Linear layer's spectral
    norm at L**(1/n_linear). Registers a SpectralCap weight-parametrization on every nn.Linear found
    under net_module. Returns the number of Linear layers capped. Idempotent per layer is NOT
    guaranteed -- call once per built net."""
    linears = [m for m in net_module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return 0
    cap = float(L) ** (1.0 / len(linears))
    for m in linears:
        register_parametrization(m, 'weight', SpectralCap(m.weight, cap, n_power))
    return len(linears)


def estimate_lipschitz(net_module: nn.Module, n_in: int, n_probe: int = 2000,
                       eps: float = 1e-3, device=None) -> float:
    """Empirical Lipschitz estimate: max over random point pairs of ||f(x+d)-f(x)|| / ||d||.
    A sanity check that the cap is actually bounding the net's input-output gain."""
    net_module.eval()
    with torch.no_grad():
        x = torch.randn(n_probe, n_in, device=device)
        d = torch.randn(n_probe, n_in, device=device); d = eps * d / d.norm(dim=1, keepdim=True)
        f0 = net_module(x); f1 = net_module(x + d)
        ratio = (f1 - f0).norm(dim=1) / d.norm(dim=1)
    return float(ratio.max())
