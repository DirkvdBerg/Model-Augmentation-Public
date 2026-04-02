"""
Reduced 4-channel LPV-LFR implementation for the dual-gantry model.
"""

from lpv_lfr_baseline.svd.lfr_svd_block import LFRBaselineBlock, LFRReducedBlock
from lpv_lfr_baseline.svd.lfr_svd_forward import lfr_forward
from lpv_lfr_baseline.svd.lfr_svd_reduction import (
    GMatrixReduced,
    G_reduced,
    build_reduced_G_matrix,
)
from lpv_lfr_baseline.svd.lfr_svd_simulate import SimResult, rk4_step, simulate

__all__ = [
    "GMatrixReduced",
    "G_reduced",
    "LFRBaselineBlock",
    "LFRReducedBlock",
    "SimResult",
    "build_reduced_G_matrix",
    "lfr_forward",
    "rk4_step",
    "simulate",
]
