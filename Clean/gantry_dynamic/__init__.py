"""gantry_dynamic: the gantry side of the clean pipeline (config, data, controller, model, OBC, training)."""
__project_origin__ = "added"

import os as _os

import model_augmentation as _ma

# One process must never mix the clean gantry_dynamic with the mainline model_augmentation
# (a launch with the repository root first on sys.path would do that silently).
_here = _os.path.normcase(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_ma_root = _os.path.normcase(_os.path.dirname(_os.path.dirname(_os.path.abspath(_ma.__file__))))
if _ma_root != _here:
    raise ImportError('gantry_dynamic in %s imported model_augmentation from %s; run the clean code with '
                      'its own folder first on sys.path (python Clean/gantry_interconnect_dynamic.py)'
                      % (_here, _ma_root))
