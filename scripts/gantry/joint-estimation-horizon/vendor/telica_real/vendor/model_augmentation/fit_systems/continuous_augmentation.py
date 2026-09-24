"""Continuous-time physical plus neural augmentation integrated as one field.

The legacy gantry graph advances the physical block with RK4 and then adds the
ANN output to the completed discrete state. That composition is globally first
order for a non-zero correction. This module evaluates the ANN output as a
normalized-state derivative at every RK4 substage, together with the physical
and augmented-state derivatives.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor

from model_augmentation.fit_systems.blocks import Block, Gantry_State_Block, Static_ANN_Block


def rk4_step(rhs: Callable[[Tensor], Tensor], state: Tensor, step: float) -> Tensor:
    """One classical RK4 step for an arbitrary autonomous right-hand side."""
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * step * k1)
    k3 = rhs(state + 0.5 * step * k2)
    k4 = rhs(state + step * k3)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


class CombinedContinuousGantryBlock(Block):
    """Advance ``[x_phys; x_aug]`` using RK4 on ``f_FP + g_theta``.

    ``ann_block`` stays registered as a top-level interconnect block for the
    existing checkpoint and diagnostic interfaces. PyTorch de-duplicates the
    shared module instance when enumerating parameters. The standalone ANN node
    is intentionally disconnected from ``xp``; this block is its only execution
    path during a state update.
    """

    def __init__(self, physical_block: Gantry_State_Block,
                 ann_block: Static_ANN_Block, route_ix: Sequence[int],
                 nxd: int, nu: int, *args, **kwargs) -> None:
        super().__init__(nz=int(nxd) + int(nu), nw=int(nxd), *args, **kwargs)
        route = tuple(int(i) for i in route_ix)
        if len(set(route)) != len(route):
            raise ValueError('route_ix contains duplicate state rows')
        if any(i < 0 or i >= nxd for i in route):
            raise ValueError('route_ix must lie inside the combined state')
        if ann_block.nz != nxd + nu or ann_block.nw != len(route):
            raise ValueError('ANN dimensions do not match combined-state routing')
        if physical_block.nx > nxd or physical_block.nu != nu:
            raise ValueError('physical block dimensions do not fit combined state')

        self.physical_block = physical_block
        self.ann_block = ann_block
        self.route_ix = route
        self.nxd = int(nxd)
        self.nu = int(nu)
        self.nx_phys = int(physical_block.nx)
        self.Ts = float(physical_block.Ts)
        self.up_sample = int(physical_block.up_sample)

    def correction_field(self, state: Tensor, u: Tensor) -> Tensor:
        """Normalized continuous-time ANN derivative on every routed state row."""
        routed = self.ann_block(torch.cat((state, u), dim=1))
        out = torch.zeros_like(state)
        if self.route_ix:
            out[:, self.route_ix, :] = routed
        return out

    def derivative(self, state: Tensor, u: Tensor) -> Tensor:
        out = self.correction_field(state, u)
        phys = self.physical_block.deriv(state[:, :self.nx_phys, :], u)
        out = out.clone()
        out[:, :self.nx_phys, :] = out[:, :self.nx_phys, :] + phys
        return out

    def forward(self, z: Tensor) -> Tensor:
        if z.ndim != 3 or z.size(1) != self.nz:
            raise ValueError('expected (batch,%d,1), got %r' % (self.nz, tuple(z.shape)))
        state, u = z[:, :self.nxd, :], z[:, self.nxd:, :]
        self.physical_block.prepare_derivative()
        h = self.Ts / self.up_sample
        for _ in range(self.up_sample):
            state = rk4_step(lambda s: self.derivative(s, u), state, h)
        return state

    # Direct-block reporting/regularization compatibility for joint estimation.
    def param_loss(self):
        if not hasattr(self.physical_block, 'param_loss'):
            return 0.0
        return self.physical_block.param_loss()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            physical = super().__getattr__('physical_block')
            if hasattr(physical, name):
                return getattr(physical, name)
            raise
