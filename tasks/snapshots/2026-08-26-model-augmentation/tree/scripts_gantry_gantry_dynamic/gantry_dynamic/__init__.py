"""gantry_dynamic: behavior-preserving package split of gantry_interconnect_dynamic.py.

Importing the package puts the repo root on sys.path so that `model_augmentation`
resolves regardless of the caller's working directory (the pre-refactor entry
file did the same with an sys.path.insert at module top).
"""
__project_origin__ = "added"

import os as _os
import sys as _sys

from .config import REPO_ROOT as _REPO_ROOT

if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
