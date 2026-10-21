"""The two helpers of the closed-loop report (D-227) that the evaluation imports."""
__project_origin__ = "added"

import copy

import torch

from model_augmentation.fit_systems.blocks import Static_ANN_Block


def _without_learned(fit_sys, free=None):
    """A deep copy of the model with the learned component removed and, if `free` is given, the
    physics block's free coordinate set to it. The model passed in is not touched (todo 7b)."""
    m = copy.deepcopy(fit_sys)
    blocks = m.hfn.connected_blocks
    ann = next((b for b in blocks if isinstance(b, Static_ANN_Block)), None)
    corr = getattr(m.hfn, 'obc_correction', None)
    with torch.no_grad():
        if ann is not None:
            ann.out_gate.zero_()
        if corr is not None:
            corr.theta_frozen.zero_()
        if free is not None:
            next(b for b in blocks if hasattr(b, 'free_params')).free_params.copy_(free)
    return m


def _true_free(phy):
    """The free coordinate at which the reduced block holds the TRUE combinations."""
    dt = phy.free_params.dtype
    c_true = type(phy).combos_of(phy.params_init.double(), float(phy.Lb)).to(dt)
    return phy.free_from_combinations(c_true)
