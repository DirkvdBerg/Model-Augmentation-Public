"""gantry_dynamic: behavior-preserving package split of gantry_interconnect_dynamic.py.

Importing the package puts the repo root on sys.path so that `model_augmentation`
resolves regardless of the caller's working directory (the pre-refactor entry
file did the same with an sys.path.insert at module top).
"""
__project_origin__ = "added"

import os as _os
import sys as _sys

# TELICA-REAL: (TR-001) the ONE import root is telica-real/vendor, never the repo root, so the
# vendored model_augmentation is the one that resolves. The original inserted REPO_ROOT here.
_VENDOR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

if _VENDOR not in _sys.path:
    _sys.path.insert(0, _VENDOR)
