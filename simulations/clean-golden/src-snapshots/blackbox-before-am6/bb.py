"""Black box M_BB: x^bb_k+1 = f_bb(x^bb_k, r_k, u^ms_k), y_hat_k = r_k + h_bb(x^bb_k), a SUBNET of Jan Hoekstra
(deepSI 0.3.29 `SS_encoder_general_hf`, nets in jan_model.py) on the servo error e = y - r. Shared parts here;
training train_bb.py, evaluation eval_bb.py, selection select_bb.py.
Conventions: code `f` is the injected force u^ms; u = (r, u^ms) and e = y - r, three stage axes each; arrays are
samples x axes."""
__project_origin__ = "added"

import os
import sys
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # Clean/
from gantry_dynamic.config import RunConfig, DATA_ROOT                             # noqa: E402
from gantry_dynamic.data import traj_dir, _record_at_fs_new                        # noqa: E402

# Encoder window n + 1 = 30 past samples of u and e (Table hyper).
NA = 30


def model_kwargs(nx, fh_width, e_width):
    """Constructor arguments of the black box.

    Shared by training and selection, so the parameter count of the search table counts the trained nets.

    Args:
        nx: state dimension n_x.
        fh_width: hidden width of f_bb and h_bb (2 tanh hidden layers each).
        e_width: hidden width of psi (2 tanh hidden layers).
    Returns:
        Keyword arguments of `SS_encoder_general_hf` and its subclasses.
    """
    from deepSI.fit_systems.encoders import default_output_net
    from jan_model import JanEncoder, JanStateNet
    fkw = dict(n_hidden_layers=2, n_nodes_per_layer=fh_width)
    return dict(nx=nx, na=NA, nb=NA, e_net=JanEncoder,
                e_net_kwargs=dict(n_hidden_layers=2, n_nodes_per_layer=e_width),
                hf_net_kwargs=dict(f_net=JanStateNet, f_net_kwargs=dict(fkw), h_net=default_output_net,
                                   h_net_kwargs=dict(fkw)))


def data_cfg(truncate_s=None):
    """Loader settings: the lowpass data set at the training rate, the RunConfig fields the loader reads."""
    return SimpleNamespace(noise='lowpass', data_root=DATA_ROOT, d=RunConfig.d, fs_orig=RunConfig.fs_orig,
                           fs_new=RunConfig.fs_new, truncate_s=truncate_s)


def load_records(files, cfg, folder=None):
    """Records of the black box at the training rate.

    Uses the grey box's loader, so both models see the same samples.

    Args:
        files: record file names.
        cfg: loader settings from `data_cfg`.
        folder: record folder; default the lowpass data set.
    Returns:
        List of (u, r, y) per record: u (N, 6) = (r, u^ms), r (N, 3) and y (N, 3), physical units.
    """
    out = []
    for fn in files:
        d = loadmat(os.path.join(folder or traj_dir(cfg), fn), squeeze_me=True)
        # y and r by the grey-box loader; u^ms by its plant-input rule (block mean over each training sample)
        f, y, r = _record_at_fs_new({**d, 'u_total': d['f_sim']}, cfg, ('r_sim',))
        out.append((np.hstack([r, f]), r, y))
    return out
