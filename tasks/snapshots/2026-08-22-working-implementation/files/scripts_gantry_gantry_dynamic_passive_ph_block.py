"""PassivePHPort -- velocity-in / force-out passive MIMO one-port (port-Hamiltonian).

__project_origin__ = "added"

Candidate route for the passivity-constrained augmentation (drift-diagnosis-status.md 5i,
"CONCRETE PHASE-1 CONSTRUCTION"). This is the STANDALONE isolation core (Phase 1); the drop-in
Static_ANN_Block wrapper + interconnect wiring is Phase 2, and framework integration into
model_augmentation/ is Phase 5. This file does NOT import or modify model_augmentation/.

Construction (continuous time):
    xi_dot = ( J - R ) gradH(xi) + G v          v = [v_X, v_Y]   (collocated X/Y velocity, INPUT)
    F      = - G^T gradH(xi)                     F = [F_X, F_Y]   (force on the X/Y rows, OUTPUT)
with H(xi) >= 0 (storage), R = L_R L_R^T >= 0 (dissipation), J = S - S^T (skew).

Passive-with-storage proof (uses only J skew, R >= 0, H >= 0 -> holds for nonlinear H,R,J too):
    dH/dt = gradH^T xi_dot = -gradH^T R gradH + gradH^T G v
    F.v   = -gradH^T G v            =>  F.v = -dH/dt - gradH^T R gradH <= -dH/dt
    => int_0^T F.v dt <= H(0) - H(T) <= H(0)                 (reset to rest: xi(0)=0 -> <= 0)
So the block may STORE and RETURN energy (spring/absorber) but not CREATE net energy (drift).

The off-diagonal (cross-axis) coupling is captured in the VELOCITY domain: both F_X and F_Y are
read from the SHARED state xi, which is driven by BOTH v_X and v_Y via the full G -> F_X depends on
v_Y and F_Y on v_X. No machine-position input is used (marginal-mode preservation: the X/Y position
pole stays at the origin because F has no position dependence).

STATEFUL (like bounded_integral_block.BoundedIntegral_ANN_Block): xi is carried across timesteps
within one rollout, stepped by explicit Euler inside the block. reset() zeros it at the start of
every rollout; xi is a LIVE tensor (BPTT-connected), detached only at reset; auto-reset on batch
change / first call. CAVEAT: explicit Euler makes the discrete energy balance passive only up to an
O(dt^2)-per-step defect; the Phase-1 energy audit QUANTIFIES it (discrete-gradient/implicit-midpoint
is the exact-discrete upgrade, theory phase). Step 1 = LINEAR config only; nonlinear H,R,J = Step 4.
"""
__project_origin__ = "added"

import torch
from torch import Tensor, nn


class PassivePHPort(nn.Module):
    """Passive port-Hamiltonian one-port: v=[v_X,v_Y] -> F=[F_X,F_Y], internal storage xi in R^m.

    Parameters
    ----------
    m : int
        Storage dimension (default 2 = truth-absorber state count; m >= 2 recommended).
    dt : float
        Integration step for xi (the pipeline Ts / up_sample). REQUIRED for the Euler step.
    n_ports : int
        Number of collocated velocity/force ports (2 = X,Y). Kept explicit for clarity/reuse.
    g_scale : float
        Scales the input/output coupling G (analogous to psi_scale in the bounded block) so the
        force can reach the needed magnitude. Passivity is scale-independent.
    nonlinear : bool
        Step 4 flag. False (Step 1) = linear/quadratic H=0.5 xi^T Q xi, constant R,J,G. True is not
        implemented yet (raises), so the linear core is verified in isolation first.
    """

    def __init__(self, m: int = 2, dt: float = None, n_ports: int = 2,
                 g_scale: float = 1.0, nonlinear: bool = False,
                 integrator: str = "midpoint", seed: int = None) -> None:
        super().__init__()
        if dt is None:
            raise ValueError("dt is required (pipeline Ts / up_sample).")
        if nonlinear:
            raise NotImplementedError(
                "nonlinear H,R,J is Step 4; Step 1 verifies the linear core in isolation.")
        if integrator not in ("euler", "midpoint"):
            raise ValueError("integrator must be 'euler' or 'midpoint'.")
        self.m = int(m)
        self.dt = float(dt)
        self.n_ports = int(n_ports)
        self.g_scale = float(g_scale)
        self.nonlinear = False
        # 'euler' = explicit Euler (Step-1; NOT discrete-passivity-tight at pipeline dt, see p1 audit);
        # 'midpoint' = implicit midpoint with midpoint force readout -> EXACTLY discrete-passive for
        # quadratic H (H(xi_{k+1})-H(xi_k) = dt[-gradH_mid^T R gradH_mid - F_k.v_k]). Default.
        self.integrator = integrator

        gen = torch.Generator().manual_seed(seed) if seed is not None else None

        def _p(*shape):
            return nn.Parameter(torch.randn(*shape, generator=gen) * 0.1)

        # Structural factors -> Q = L_Q L_Q^T >= 0, R = L_R L_R^T >= 0, J = S - S^T (skew).
        self.L_Q = _p(self.m, self.m)      # storage metric factor
        self.L_R = _p(self.m, self.m)      # dissipation factor
        self.S   = _p(self.m, self.m)      # skew generator
        self.G   = _p(self.m, self.n_ports)  # input matrix (full -> off-diagonal coupling)

        self._xi = None                    # live tensor (batch, m) or None; NOT a buffer (BPTT)

    # -- structural matrices (constant in the linear/Step-1 config) --------------------------------
    def Q(self) -> Tensor:
        return self.L_Q @ self.L_Q.t()

    def R(self) -> Tensor:
        return self.L_R @ self.L_R.t()

    def J(self) -> Tensor:
        return self.S - self.S.t()

    def _G(self) -> Tensor:
        return self.g_scale * self.G

    # -- storage function and its gradient ---------------------------------------------------------
    def H(self, xi: Tensor) -> Tensor:
        """Storage H(xi) = 0.5 xi^T Q xi >= 0. xi: (batch, m) -> (batch,)."""
        Qx = xi @ self.Q().t()             # (batch, m)
        return 0.5 * (xi * Qx).sum(dim=1)

    def gradH(self, xi: Tensor) -> Tensor:
        """grad_xi H = Q xi. xi: (batch, m) -> (batch, m)."""
        return xi @ self.Q().t()

    # -- rollout control ---------------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the carried xi. Call at the start of every rollout (window/sim)."""
        self._xi = None

    def current_storage(self) -> Tensor:
        """H(xi) for the currently carried state (0 if not yet stepped)."""
        if self._xi is None:
            return torch.zeros(1)
        return self.H(self._xi)

    # -- one timestep ------------------------------------------------------------------------------
    def forward(self, v: Tensor) -> Tensor:
        """Step xi one timestep and return the collocated force F.

        v : (batch, n_ports)  collocated velocity  ->  F : (batch, n_ports) force.
        First call (or batch change): xi seeded to 0. At rest (v=0) F = 0.

        integrator='euler'   : F from xi_k, then xi_{k+1} = xi_k + dt[(J-R)Q xi_k + G v]  (O(dt) defect)
        integrator='midpoint': implicit midpoint, F from xi_mid -> EXACT discrete passivity (quadratic H).
        """
        assert v.dim() == 2 and v.size(1) == self.n_ports, \
            f"v must be (batch, {self.n_ports}), got {tuple(v.shape)}"
        batch = v.size(0)
        xi = self._xi
        if xi is None or xi.shape[0] != batch:
            xi = torch.zeros(batch, self.m, dtype=v.dtype, device=v.device)

        Q = self.Q()
        G = self._G()
        A = (self.J() - self.R()) @ Q                # xi_dot = A xi + G v,  A = (J-R)Q  (m, m)

        if self.integrator == "euler":
            gH = xi @ Q.t()                          # gradH(xi_k)
            F = -(gH @ G)                            # -G^T gradH(xi_k)
            xi_next = xi + self.dt * (xi @ A.t() + v @ G.t())
        else:  # midpoint (default): (I - dt/2 A) xi_{k+1} = (I + dt/2 A) xi_k + dt G v
            I = torch.eye(self.m, dtype=v.dtype, device=v.device)
            M1 = I - 0.5 * self.dt * A
            M2 = I + 0.5 * self.dt * A
            rhs = xi @ M2.t() + self.dt * (v @ G.t())            # (batch, m)
            xi_next = torch.linalg.solve(M1, rhs.t()).t()        # (batch, m)
            xi_mid = 0.5 * (xi + xi_next)
            gH_mid = xi_mid @ Q.t()                              # gradH at the midpoint
            F = -(gH_mid @ G)                                    # -G^T gradH(xi_mid)

        self._xi = xi_next                           # carry (live tensor -> BPTT connects)
        return F
