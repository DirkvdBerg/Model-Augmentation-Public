"""Two helpers of the evaluation: the model without its learned part, and the coordinate of the true parameters."""
__project_origin__ = "added"

import copy

import torch

from model_augmentation.fit_systems.blocks import Static_ANN_Block


def _without_learned(fit_sys, free=None):
    """Baseline-only copy of a model (M_LPV in the evaluation).

    The network gate and theta_aux set to zero, optionally at another beta, so the copy simulates the baseline alone.

    Args:
        fit_sys: the augmented model; free: (10,) beta, or None to keep the current one.
    Returns:
        The modified copy.
    """
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
    """Training coordinate of the true parameters.

    beta_star = beta(theta_base(theta^0)), the coordinate at which the baseline holds the true combinations.

    Args:
        phy: the identifiable baseline block.
    Returns:
        (10,) beta_star.
    """
    dt = phy.free_params.dtype
    c_true = type(phy).combos_of(phy.params_init.double(), float(phy.Lb)).to(dt)
    return phy.free_from_combinations(c_true)
