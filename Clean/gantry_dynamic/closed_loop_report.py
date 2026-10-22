"""Two helpers of the evaluation: the model without its learned part, and the coordinate of the true parameters."""
__project_origin__ = "added"

import copy

import torch

from model_augmentation.fit_systems.blocks import Static_ANN_Block


def _without_learned(fit_sys, free=None):
    """Return a copy with the network and OBC correction disabled, optionally at another training coordinate."""
    m = copy.deepcopy(fit_sys)
    blocks = m.hfn.connected_blocks
    ann = next(b for b in blocks if isinstance(b, Static_ANN_Block))
    corr = getattr(m.hfn, 'obc_correction', None)
    with torch.no_grad():
        ann.out_gate.zero_()
        if corr is not None:
            corr.theta_frozen.zero_()
        if free is not None:
            next(b for b in blocks if hasattr(b, 'free_params')).free_params.copy_(free)
    return m


def _true_free(phy):
    """The training coordinate beta at which the physics block holds the true combinations."""
    dt = phy.free_params.dtype
    c_true = type(phy).combos_of(phy.params_init.double(), float(phy.Lb)).to(dt)
    return phy.free_from_combinations(c_true)
