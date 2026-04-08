"""
lpv_lfr_baseline
================
Standalone implementation of the dual-gantry CT LPV-LFR baseline.

Derived from: LPV/LFR-derivation-supervisor.tex
Method: resolve-and-retain (see docs/lfr-baseline-implementation-method.md)
No dependencies on model_augmentation/.

Package structure
-----------------
core/    — pure LFR physics (physics, lfr_matrices, lfr_forward, lfr_simulate)
blocks/  — Jan-compatible Block wrappers (lfr_block, lfr_param_block, lfr_fit_system)
svd/     — SVD-reduced LFR variant (n_z: 6 → 4)
scripts/ — runnable entry points and data utilities
tests/   — integration and compatibility tests
docs/    — design notes and LaTeX documentation
"""

# Re-export public API for backwards compatibility
from lpv_lfr_baseline.core.physics import (  # noqa: F401
    M0, M1, M2, K, C, P, fs, ts, build_M,
)
from lpv_lfr_baseline.core.lfr_matrices import G, GMatrix  # noqa: F401
from lpv_lfr_baseline.core.lfr_forward import lfr_forward  # noqa: F401
from lpv_lfr_baseline.core.lfr_simulate import (  # noqa: F401
    rk4_step, simulate, simulate_frozen, SimResult,
)
from lpv_lfr_baseline.blocks.lfr_block import LFRBaselineBlock  # noqa: F401
from lpv_lfr_baseline.blocks.lfr_param_block import ParameterizedLFRBlock  # noqa: F401
from lpv_lfr_baseline.blocks.lfr_fit_system import LFRFitSystem  # noqa: F401
