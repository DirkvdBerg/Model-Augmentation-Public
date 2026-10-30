"""The gantry side of the training code: configuration, data, controller, model, OBC and training."""
__project_origin__ = "added"

import os as _os

import model_augmentation as _ma

# this package must run with the model_augmentation of the same folder; a launch with the repository root first
# on sys.path would mix it with the mainline one without an error
_here = _os.path.normcase(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_ma_root = _os.path.normcase(_os.path.dirname(_os.path.dirname(_os.path.abspath(_ma.__file__))))
if _ma_root != _here:
    raise ImportError('gantry_dynamic in %s imported model_augmentation from %s; run the clean code with '
                      'its own folder first on sys.path (python Clean/gantry_interconnect_dynamic.py)'
                      % (_here, _ma_root))
